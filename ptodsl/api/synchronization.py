from typing import Sequence

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


def _resolve_sync_op_attr(sync_op):
    resolved = _resolve_sync_op(sync_op)
    if isinstance(resolved, _pto.SyncOpType):
        return _pto.SyncOpTypeAttr.get(resolved)
    return resolved


def _resolve_event_id(event_id):
    if isinstance(event_id, int):
        if event_id < 0 or event_id > 7:
            raise ValueError(f"event_id must be in range [0, 7], got {event_id}.")
        return getattr(_pto, f"EVENT_ID{event_id}")
    return event_id


def record_event(record_op, wait_op, event_id: int | Sequence[int] = 0):
    if not isinstance(event_id, int):
        for eid in event_id:
            _pto.record_event(
                _resolve_sync_op(record_op),
                _resolve_sync_op(wait_op),
                _resolve_event_id(eid),
            )
    else:
        _pto.record_event(
            _resolve_sync_op(record_op),
            _resolve_sync_op(wait_op),
            _resolve_event_id(event_id),
        )


def wait_event(record_op, wait_op, event_id: int | Sequence[int] = 0):
    if not isinstance(event_id, int):
        for eid in event_id:
            _pto.wait_event(
                _resolve_sync_op(record_op),
                _resolve_sync_op(wait_op),
                _resolve_event_id(eid),
            )
    else:
        _pto.wait_event(
            _resolve_sync_op(record_op),
            _resolve_sync_op(wait_op),
            _resolve_event_id(event_id),
        )


def record_wait_pair(record_op, wait_op, event_id: int | Sequence[int] = 0):
    record = _resolve_sync_op(record_op)
    wait = _resolve_sync_op(wait_op)
    event = _resolve_event_id(event_id)
    _pto.record_event(record, wait, event)
    _pto.wait_event(record, wait, event)


def barrier(sync_op):
    _pto.barrier(_resolve_sync_op(sync_op))


def barrier_sync(sync_op):
    _pto.barrier_sync(_resolve_sync_op_attr(sync_op))


__all__ = ["record_event", "wait_event", "record_wait_pair", "barrier", "barrier_sync"]
