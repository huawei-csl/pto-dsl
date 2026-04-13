import inspect

from mlir.dialects import func, pto as _pto
from mlir.ir import Attribute, Context, InsertionPoint, Location, Module, UnitAttr

from ..api.scalar import wrap_value


# For the inner decorators to be clean for the user visible API `pto.func(kernel='cube')`
# with no reference to module, we need this:
_CURRENT = None


class FuncRef:
    def __init__(self, sym_name):
        self.sym_name = sym_name


def _resolve_meta(meta_fn):
    values = meta_fn()
    if not isinstance(values, dict):
        raise ValueError(
            "`meta_data()` must return a dict of named symbols to MLIR/PTO types."
        )
    return dict(values)


def _resolve_arg_types(signature, meta_map):
    arg_types = []
    for param in signature.parameters.values():
        annot = param.annotation
        if isinstance(annot, str):
            if annot not in meta_map:
                raise ValueError(f"Unknown annotation '{annot}'.")
            arg_types.append(meta_map[annot])
        elif annot is inspect._empty:
            raise ValueError(f"Missing annotation for argument '{param.name}'.")
        else:
            arg_types.append(annot)
    return arg_types


def _resolve_ret_types(signature, meta_map):
    ret_annot = signature.return_annotation
    if ret_annot in (inspect._empty, None):
        return []
    if isinstance(ret_annot, str):
        if ret_annot not in meta_map:
            raise ValueError(f"Unknown return annotation '{ret_annot}'.")
        return [meta_map[ret_annot]]
    if isinstance(ret_annot, (list, tuple)):
        out = []
        for elem in ret_annot:
            out.append(meta_map[elem] if isinstance(elem, str) else elem)
        return out
    return [ret_annot]


def _has_func_return(block):
    last_name = None
    for op in block.operations:
        last_name = op.operation.name
    return last_name == "func.return"


def _inject_globals(fn, values):
    old = {}
    for name, value in values.items():
        old[name] = fn.__globals__.get(name, None)
        fn.__globals__[name] = value
    return old


def _restore_globals(fn, old, names):
    for name in names:
        if old[name] is None and name in fn.__globals__:
            del fn.__globals__[name]
        else:
            fn.__globals__[name] = old[name]


def _define(module, ctx, meta_map, fn, *, name=None, entry=False, kernel=None):
    sig = inspect.signature(fn)
    arg_types = _resolve_arg_types(sig, meta_map)
    ret_types = _resolve_ret_types(sig, meta_map)
    fn_name = name or fn.__name__
    fn_ty = func.FunctionType.get(arg_types, ret_types)

    with InsertionPoint(module.body):
        ir_func = func.FuncOp(fn_name, fn_ty)

    if entry:
        ir_func.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
    if kernel is not None:
        ir_func.operation.attributes["pto.kernel_kind"] = Attribute.parse(
            f"#pto.kernel_kind<{kernel}>"
        )

    block = ir_func.add_entry_block()
    with InsertionPoint(block):
        wrapped_args = [wrap_value(arg) for arg in block.arguments]
        old = _inject_globals(fn, meta_map)
        try:
            fn(*wrapped_args)
        finally:
            _restore_globals(fn, old, meta_map.keys())

        if not ret_types and not _has_func_return(block):
            func.ReturnOp([])

    return FuncRef(fn_name)


def ir_func(fn=None, *, name=None, kernel=None):
    entry = kernel is None

    def decorator(fn):
        if _CURRENT is None:
            raise RuntimeError(
                "`pto.func` can only be used inside `@to_ir_module(..., module=True)`."
            )
        return _define(
            _CURRENT["module"],
            _CURRENT["ctx"],
            _CURRENT["meta_map"],
            fn,
            name=name,
            entry=entry,
            kernel=kernel,
        )

    if fn is not None:
        return decorator(fn)

    return decorator


def to_ir_module(*, meta_data, module=False):
    def decorator(fn):
        global _CURRENT

        with Context() as ctx, Location.unknown():
            _pto.register_dialect(ctx, load=True)
            meta_map = _resolve_meta(meta_data)
            ir_module = Module.create()

            if module:
                if inspect.signature(fn).parameters:
                    raise ValueError(
                        "`module=True` expects a zero-argument builder function."
                    )
                old = _inject_globals(fn, meta_map)
                prev = _CURRENT
                _CURRENT = {"ctx": ctx, "module": ir_module, "meta_map": meta_map}
                try:
                    fn()
                finally:
                    _CURRENT = prev
                    _restore_globals(fn, old, meta_map.keys())
            else:
                _define(ir_module, ctx, meta_map, fn)

            ir_module.operation.verify()
            return ir_module

    return decorator


__all__ = ["FuncRef", "ir_func", "to_ir_module"]
