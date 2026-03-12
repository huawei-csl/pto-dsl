import inspect

from mlir.dialects import func, pto as _pto
from mlir.ir import Context, InsertionPoint, Location, Module

from ..api.scalar import wrap_value
from .ast_lowering import lower_function


def _resolve_meta(meta_fn):
    values = meta_fn()
    if not isinstance(values, dict):
        raise ValueError("`meta_data()` must return a dict of named symbols to MLIR/PTO types.")
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
            if isinstance(elem, str):
                out.append(meta_map[elem])
            else:
                out.append(elem)
        return out
    return [ret_annot]


def _has_func_return(block):
    last_name = None
    for op in block.operations:
        last_name = op.operation.name
    return last_name == "func.return"


def to_ir_module(*, meta_data):
    def decorator(fn):
        sig = inspect.signature(fn)

        with Context() as ctx, Location.unknown():
            _pto.register_dialect(ctx, load=True)
            meta_map = _resolve_meta(meta_data)
            arg_types = _resolve_arg_types(sig, meta_map)
            ret_types = _resolve_ret_types(sig, meta_map)
            module = Module.create()
            fn_ty = func.FunctionType.get(arg_types, ret_types)

            with InsertionPoint(module.body):
                ir_func = func.FuncOp(fn.__name__, fn_ty)
                entry = ir_func.add_entry_block()

            with InsertionPoint(entry):
                wrapped_args = [wrap_value(arg) for arg in entry.arguments]
                _, terminated = lower_function(fn, wrapped_args, meta_map)

                if not ret_types and not terminated and not _has_func_return(entry):
                    func.ReturnOp([])

            module.operation.verify()
            return module

    return decorator


__all__ = ["to_ir_module"]
