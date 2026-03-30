from mlir.dialects import pto as mlir_pto
from mlir.ir import Context, IndexType, Location

from ptodsl import micro, pto, to_ir_module
from ptodsl.api._micro_registry import MICRO_OPS
from tests._vpto_manifest import load_vpto_manifest


def test_micro_type_builders_support_memory_space_strings():
    with Context() as ctx, Location.unknown():
        mlir_pto.register_dialect(ctx, load=True)

        assert str(pto.PtrType(pto.float32, memory_space="VEC")) == "!pto.ptr<f32, ub>"
        assert str(pto.VRegType(64, pto.float32)) == "!pto.vreg<64xf32>"
        assert str(pto.MaskType()) == "!pto.mask"
        assert str(pto.AlignType()) == "!pto.align"


def test_pure_micro_kernel_emits_vpto_ops():
    def meta_data():
        return {
            "ptr_t": pto.PtrType(pto.float32, memory_space="VEC"),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def micro_kernel(src: "ptr_t", dst: "ptr_t", offset: "index_t") -> None:
        mask = micro.pset_b32(pto.MaskType(), "PAT_ALL")
        vec = micro.vlds(pto.VRegType(64, pto.float32), src, offset)
        micro.vsts(vec, dst, offset, mask)

    text = str(micro_kernel)

    assert "func.func @micro_kernel" in text
    assert "pto.pset_b32" in text
    assert "pto.vlds" in text
    assert "pto.vsts" in text
    assert "!pto.vreg<64xf32>" in text


def test_mixed_tile_and_micro_kernel_emit_together():
    def meta_data():
        dtype = pto.float32
        return {
            "ptr_t": pto.PtrType(dtype, memory_space="VEC"),
            "index_t": IndexType.get(),
            "tile_type": pto.TileBufType(
                shape=[1, 64],
                valid_shape=[1, 64],
                dtype=dtype,
                memory_space="VEC",
            ),
        }

    @to_ir_module(meta_data=meta_data)
    def mixed_kernel(src: "ptr_t", dst: "ptr_t", offset: "index_t") -> None:
        pto.alloc_tile(tile_type)
        mask = pto.pset_b32(pto.MaskType(), "PAT_ALL")
        vec = pto.vlds(pto.VRegType(64, pto.float32), src, offset)
        pto.vsts(vec, dst, offset, mask)

    text = str(mixed_kernel)

    assert "pto.alloc_tile" in text
    assert "pto.vlds" in text
    assert "pto.vsts" in text


def test_micro_registry_matches_manifest_inventory():
    manifest = load_vpto_manifest()
    implemented = {
        op["mnemonic"] for op in manifest["ops"] if op.get("status") == "implemented"
    }

    assert set(MICRO_OPS) == implemented
    assert all(hasattr(micro, name) for name in implemented)
    assert all(hasattr(pto, name) for name in implemented)
