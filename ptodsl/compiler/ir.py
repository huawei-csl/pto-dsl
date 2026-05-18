import inspect

from mlir.dialects import func, pto as _pto
from mlir.ir import Attribute, Context, InsertionPoint, Location, Module, UnitAttr

from ..api.scalar import wrap_value
from ..api.type_def import _LazyType, _materialize
from ..utils.codegen import get_user_code_loc


# For the inner decorators to be clean for the user visible API `pto.func(kernel='cube')`
# with no reference to module, we need this:
_CURRENT = None


class _KernelIR:
    """Pairs an ``mlir.ir.Module`` with the original Python kernel function.

    ``to_ir_module`` returns one of these so that downstream tools (e.g.
    caller-cpp generation) can introspect the function signature even after
    the decorator has replaced the Python name with the compiled IR.

    The object delegates ``__str__`` and ``__repr__`` to the inner module so
    that existing code comparing IR text continues to work.  Equality is also
    forwarded, as is ``isinstance(obj, mlir.ir.Module)``-style duck-typing via
    the ``ir_module`` attribute.
    """

    def __init__(self, ir_module: Module, source_fn):
        self.ir_module = ir_module
        self._source_fn = source_fn

    # --- delegation helpers ---

    def __str__(self):
        return str(self.ir_module)

    def __repr__(self):
        return repr(self.ir_module)

    def __eq__(self, other):
        if isinstance(other, _KernelIR):
            return str(self) == str(other)
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    # Delegate MLIR module attributes so existing code using
    # ``kernel.operation``, ``kernel.body``, etc. continues to work.
    def __getattr__(self, name):
        return getattr(self.ir_module, name)


class FuncRef:
    def __init__(self, sym_name):
        self.sym_name = sym_name


def _collect_lazy_globals(fn):
    """Return a dict of all names visible to *fn* that are _LazyType instances.

    Scans both the function's module globals and its closure variables.
    """
    lazy = {}

    # Module-level globals referenced by the function
    for name, value in fn.__globals__.items():
        if isinstance(value, _LazyType):
            lazy[name] = value

    # Closure variables (for nested / parameterised builders)
    if fn.__code__.co_freevars and fn.__closure__:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, _LazyType):
                lazy[name] = value

    return lazy


def _materialize_lazy_globals(fn):
    """Materialize all _LazyType values visible to *fn* and return a name→type map.

    Must be called inside an active MLIR Context.
    """
    lazy_map = _collect_lazy_globals(fn)
    return {name: lazy.materialize() for name, lazy in lazy_map.items()}


def _resolve_arg_types(signature, meta_map):
    arg_types = []
    for param in signature.parameters.values():
        annot = param.annotation
        if isinstance(annot, str):
            if annot not in meta_map:
                raise ValueError(f"Unknown annotation '{annot}'.")
            arg_types.append(meta_map[annot])
        elif isinstance(annot, _LazyType):
            arg_types.append(_materialize(annot))
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
    if isinstance(ret_annot, _LazyType):
        return [_materialize(ret_annot)]
    if isinstance(ret_annot, (list, tuple)):
        out = []
        for elem in ret_annot:
            if isinstance(elem, str):
                out.append(meta_map[elem])
            elif isinstance(elem, _LazyType):
                out.append(_materialize(elem))
            else:
                out.append(elem)
        return out
    return [ret_annot]


def _has_func_return(block):
    last_name = None
    for op in block.operations:
        last_name = op.operation.name
    return last_name == "func.return"


def _get_globals(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn.__globals__


def _inject_globals(fn, values):
    globs = _get_globals(fn)
    old = {}
    for name, value in values.items():
        old[name] = globs.get(name, None)
        globs[name] = value
    return old


def _restore_globals(fn, old, names):
    globs = _get_globals(fn)
    for name in names:
        if old[name] is None and name in globs:
            del globs[name]
        else:
            globs[name] = old[name]


def _define(module, ctx, meta_map, fn, *, name=None, entry=False, kernel=None):
    sig = inspect.signature(fn)
    arg_types = _resolve_arg_types(sig, meta_map)
    ret_types = _resolve_ret_types(sig, meta_map)
    fn_name = name or fn.__name__
    fn_ty = func.FunctionType.get(arg_types, ret_types)

    fn_file = inspect.getsourcefile(fn)
    fn_line = inspect.getsourcelines(fn)[1]
    with InsertionPoint(module.body), Location.file(fn_file, fn_line, 0):
        ir_func = func.FuncOp(fn_name, fn_ty)

    if entry:
        ir_func.operation.attributes["pto.entry"] = UnitAttr.get(ctx)
    if kernel is not None:
        ir_func.operation.attributes["pto.kernel_kind"] = Attribute.parse(
            f"#pto.kernel_kind<{kernel}>"
        )

    block = ir_func.add_entry_block()
    with InsertionPoint(block), Location.file(fn_file, fn_line, 0):
        wrapped_args = [wrap_value(arg) for arg in block.arguments]
        old = _inject_globals(fn, meta_map)
        try:
            fn(*wrapped_args)
        finally:
            _restore_globals(fn, old, meta_map.keys())

        if not ret_types and not _has_func_return(block):
            func.ReturnOp([])

    # When building a multi-function module, record the entry function's
    # metadata so that JitWrapper can discover the signature for caller.cpp.
    if entry and _CURRENT is not None:
        _CURRENT["entry_name"] = fn_name
        _CURRENT["entry_sig"] = sig
        _CURRENT["entry_arg_types"] = arg_types

    return FuncRef(fn_name)


def ir_func(fn=None, *, name=None, kernel=None):
    entry = kernel is None

    def decorator(fn):
        if _CURRENT is None:
            raise RuntimeError(
                "`pto.func` can only be used inside `@to_ir_module(module=True)`."
            )
        ref = _define(
            _CURRENT["module"],
            _CURRENT["ctx"],
            _CURRENT["meta_map"],
            fn,
            name=name,
            entry=entry,
            kernel=kernel,
        )
        # Record the entry function so to_ir_module can attach it as _source_fn.
        if entry:
            _CURRENT["entry_fn"] = fn
        return ref

    if fn is not None:
        return decorator(fn)

    return decorator


def to_ir_module(fn=None, *, module=False):
    """Decorator that compiles a kernel function (or module builder) to an MLIR module.

    Usage::

        @to_ir_module
        def my_kernel(arg: ptr_type) -> None:
            ...

        # or with module=True for multi-function modules:
        @to_ir_module(module=True)
        def my_module():
            ...
    """

    def decorator(fn):
        global _CURRENT, _LAST_ENTRY_META
        _LAST_ENTRY_META = None

        with Context() as ctx, get_user_code_loc():
            _pto.register_dialect(ctx, load=True)
            meta_map = _materialize_lazy_globals(fn)
            ir_module = Module.create()

            if module:
                if inspect.signature(fn).parameters:
                    raise ValueError(
                        "`module=True` expects a zero-argument builder function."
                    )
                old = _inject_globals(fn, meta_map)
                prev = _CURRENT
                _CURRENT = {
                    "ctx": ctx,
                    "module": ir_module,
                    "meta_map": meta_map,
                    "entry_fn": None,
                }
                try:
                    fn()
                    # Capture entry metadata before _CURRENT is restored.
                    _LAST_ENTRY_META = {
                        "entry_name": _CURRENT.get("entry_name"),
                        "entry_sig": _CURRENT.get("entry_sig"),
                        "entry_arg_types": _CURRENT.get("entry_arg_types"),
                    }
                finally:
                    entry_fn = _CURRENT.get("entry_fn")
                    _CURRENT = prev
                    _restore_globals(fn, old, meta_map.keys())
                # For module=True, prefer the entry function for caller-gen.
                source_fn = entry_fn if entry_fn is not None else fn
            else:
                _define(ir_module, ctx, meta_map, fn)
                source_fn = fn

            ir_module.operation.verify()
            # Attach the original Python function so callers can introspect it
            # (e.g. to generate caller.cpp from the signature).
            return _KernelIR(ir_module, source_fn)

    # Support both @to_ir_module and @to_ir_module(module=True)
    if fn is not None:
        return decorator(fn)
    return decorator


__all__ = ["FuncRef", "get_last_entry_meta", "ir_func", "to_ir_module"]
