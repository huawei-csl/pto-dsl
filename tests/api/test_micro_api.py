from types import SimpleNamespace

from mlir.dialects import pto as mlir_pto
from mlir.ir import IndexType

from ptodsl import micro, pto, to_ir_module
from ptodsl.api import micro as micro_api
from tests._vpto_manifest import load_vpto_manifest


IMPLEMENTED_MICRO_OPS = sorted(
    op["mnemonic"]
    for op in load_vpto_manifest()["ops"]
    if op.get("status") == "implemented"
)


class _Box:
    def __init__(self, raw):
        self.raw = raw


def test_manifest_driven_micro_inventory_matches_public_exports():
    assert set(IMPLEMENTED_MICRO_OPS) == set(micro.MICRO_OPS)
    assert set(IMPLEMENTED_MICRO_OPS).issubset(set(micro.__all__))
    assert set(IMPLEMENTED_MICRO_OPS).issubset(set(pto.__all__))


def test_every_manifest_micro_op_has_a_callable_wrapper():
    for name in IMPLEMENTED_MICRO_OPS:
        wrapper = getattr(micro, name)

        assert callable(wrapper)
        assert getattr(pto, name) is wrapper
        assert hasattr(mlir_pto, name)

        if name == "barrier":
            assert wrapper is micro_api._micro_barrier
        else:
            assert wrapper.__name__ == name
            assert f"`pto.{name}`" in (wrapper.__doc__ or "")


def test_barrier_normalizes_pipe_names(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        micro_api._pto,
        "PIPE",
        SimpleNamespace(PIPE_VECTOR="PIPE_VECTOR"),
        raising=False,
    )
    monkeypatch.setattr(
        micro_api._pto,
        "PipeAttr",
        SimpleNamespace(get=lambda value: f"pipe:{value}"),
    )
    monkeypatch.setattr(
        micro_api._pto,
        "barrier",
        lambda op, loc=None, ip=None: seen.setdefault("call", (op, loc, ip)),
    )

    micro.barrier("pipe_vector")

    assert seen["call"] == ("pipe:PIPE_VECTOR", None, None)


def test_selected_micro_wrappers_emit_ir_and_accept_objects_with_raw_values():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32, space="VEC"),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def kernel(src: "ptr_t", dst: "ptr_t", offset: "index_t") -> None:
        mask = micro.pset_b32(pto.MaskType(), "PAT_ALL")
        vec = micro.vlds(pto.VRegType(64, pto.float32), _Box(src), offset)
        micro.vsts(vec, _Box(dst), offset, _Box(mask))

    text = str(kernel)

    assert "pto.pset_b32" in text
    assert "pto.vlds" in text
    assert "pto.vsts" in text
