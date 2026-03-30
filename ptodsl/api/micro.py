from mlir.dialects import pto as _pto

from ._micro_registry import MICRO_OPS
from .scalar import Value


def _unwrap(value):
    if isinstance(value, Value):
        return _unwrap(value.raw)
    if hasattr(value, "raw"):
        return _unwrap(value.raw)
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_unwrap(v) for v in value)
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    return value


def _micro_barrier(op, *, loc=None, ip=None):
    if isinstance(op, str):
        normalized = op.strip().upper()
        if normalized.startswith("PIPE"):
            op = _pto.PipeAttr.get(getattr(_pto.PIPE, normalized))
    return _pto.barrier(_unwrap(op), loc=loc, ip=ip)


def _make_wrapper(name):
    if name == "barrier":
        return _micro_barrier

    op = getattr(_pto, name, None)
    if op is None:
        raise AttributeError(f"mlir.dialects.pto has no builder for '{name}'")

    def _wrapper(*args, **kwargs):
        return op(*(_unwrap(arg) for arg in args), **{k: _unwrap(v) for k, v in kwargs.items()})

    _wrapper.__name__ = name
    _wrapper.__qualname__ = name
    _wrapper.__doc__ = f"Emit `pto.{name}`."
    return _wrapper


for _mnemonic in MICRO_OPS:
    globals()[_mnemonic] = _make_wrapper(_mnemonic)


__all__ = ["MICRO_OPS", *MICRO_OPS]
