from types import SimpleNamespace

from ptodsl.api import synchronization as sync


def test_barrier_normalizes_string_sync_ops(monkeypatch):
    seen = {}

    monkeypatch.setattr(sync._pto, "TDMA", "TDMA", raising=False)
    monkeypatch.setattr(sync._pto, "barrier", lambda op: seen.setdefault("op", op))

    sync.barrier("dma")

    assert seen["op"] == "TDMA"


def test_barrier_sync_normalizes_string_sync_ops(monkeypatch):
    seen = {}

    monkeypatch.setattr(sync._pto, "TVEC", sync._pto.SyncOpType.TVEC, raising=False)
    monkeypatch.setattr(
        sync._pto,
        "SyncOpTypeAttr",
        SimpleNamespace(get=lambda value: f"sync:{value}"),
    )
    monkeypatch.setattr(
        sync._pto, "barrier_sync", lambda op: seen.setdefault("op", op)
    )

    sync.barrier_sync("vec")

    assert seen["op"] == f"sync:{sync._pto.SyncOpType.TVEC}"


def test_record_event_expands_sequence_event_ids(monkeypatch):
    calls = []

    monkeypatch.setattr(sync._pto, "TDMA", "TDMA", raising=False)
    monkeypatch.setattr(sync._pto, "TVEC", "TVEC", raising=False)
    monkeypatch.setattr(sync._pto, "EVENT_ID1", "EVENT_ID1", raising=False)
    monkeypatch.setattr(sync._pto, "EVENT_ID2", "EVENT_ID2", raising=False)
    monkeypatch.setattr(sync._pto, "record_event", lambda *args: calls.append(args))

    sync.record_event("dma", "vec", [1, 2])

    assert calls == [
        ("TDMA", "TVEC", "EVENT_ID1"),
        ("TDMA", "TVEC", "EVENT_ID2"),
    ]


def test_wait_event_rejects_invalid_event_ids(monkeypatch):
    monkeypatch.setattr(sync._pto, "TDMA", "TDMA", raising=False)
    monkeypatch.setattr(sync._pto, "TVEC", "TVEC", raising=False)

    try:
        sync.wait_event("dma", "vec", 8)
    except ValueError as exc:
        assert "event_id must be in range [0, 7]" in str(exc)
    else:
        raise AssertionError("wait_event accepted an out-of-range event_id")


def test_record_wait_pair_calls_record_and_wait_once(monkeypatch):
    calls = []

    monkeypatch.setattr(sync._pto, "TDMA", "TDMA", raising=False)
    monkeypatch.setattr(sync._pto, "TVEC", "TVEC", raising=False)
    monkeypatch.setattr(sync._pto, "EVENT_ID0", "EVENT_ID0", raising=False)
    monkeypatch.setattr(
        sync._pto, "record_event", lambda *args: calls.append(("record", args))
    )
    monkeypatch.setattr(
        sync._pto, "wait_event", lambda *args: calls.append(("wait", args))
    )

    sync.record_wait_pair("dma", "vec")

    assert calls == [
        ("record", ("TDMA", "TVEC", "EVENT_ID0")),
        ("wait", ("TDMA", "TVEC", "EVENT_ID0")),
    ]
