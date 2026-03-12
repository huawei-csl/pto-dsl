from mlir.dialects import pto as _pto


def _resolve_sync_op(sync_op):
    if isinstance(sync_op, str):
        normalized = sync_op.strip().upper()
        if not normalized.startswith("T"):
            normalized = f"T{normalized}"
        try:
            return getattr(_pto, normalized)
        except AttributeError as exc:
            raise ValueError(f"Unsupported sync op type '{sync_op}'.") from exc
    return sync_op


def barrier(sync_op):
    _pto.barrier(_resolve_sync_op(sync_op))


__all__ = ["barrier"]
