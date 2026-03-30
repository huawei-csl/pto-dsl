import pytest
from mlir.ir import IndexType

from ptodsl import pto, scalar as s, tile, to_ir_module


const = s.const


def test_pto_namespace_exports_pythonic_helpers():
    exports = set(pto.__all__)
    assert {
        "ptr",
        "make_tensor",
        "make_tile_buffer",
        "TensorView",
        "TileBuffer",
        "barrier_sync",
    }.issubset(exports)


def test_ptr_and_make_tile_buffer_match_low_level_type_builders(mlir_ctx):
    assert str(pto.ptr(pto.float32)) == str(pto.PtrType(pto.float32))
    assert str(pto.ptr(pto.float32, space="VEC")) == str(
        pto.PtrType(pto.float32, memory_space="VEC")
    )

    tile_spec = pto.make_tile_buffer(pto.float32, [32, 32], space="VEC")
    assert str(tile_spec.raw_type) == str(
        pto.TileBufType(
            shape=[32, 32],
            valid_shape=[32, 32],
            dtype=pto.float32,
            memory_space="VEC",
        )
    )


def test_make_tensor_requires_dtype_when_type_is_omitted():
    with pytest.raises(TypeError, match="requires `dtype=`"):
        pto.make_tensor("ptr", shape=[32, 32])


def test_tensor_view_slice_requires_static_shape_for_dynamic_sizes():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    def kernel(src: "ptr_t", total_elements: "index_t") -> None:
        view = pto.make_tensor(src, shape=[total_elements], dtype=pto.float32)
        view.slice([0], [total_elements])

    with pytest.raises(TypeError, match="requires `static_shape=`"):
        to_ir_module(meta_data=meta_data)(kernel)


def test_pythonic_pto_wrappers_emit_tensor_and_tile_ops():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def kernel(src: "ptr_t", dst: "ptr_t", rows: "index_t", cols: "index_t") -> None:
        src_view = pto.make_tensor(src, shape=[rows, cols], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[rows, cols], dtype=pto.float32)
        src_tile = src_view.slice([0, 0], [32, 32])
        dst_tile = dst_view.slice([0, 0], [32, 32])

        with pto.vector_section():
            tile_spec = pto.make_tile_buffer(pto.float32, [32, 32], space="VEC")
            tmp = tile_spec.alloc()
            tmp.load_from(src_tile)
            tile.add(tmp, tmp, tmp)
            tmp.store_to(dst_tile)

    text = str(kernel)

    assert "pto.make_tensor_view" in text
    assert "pto.partition_view" in text
    assert "pto.alloc_tile" in text
    assert "pto.tload" in text
    assert "pto.tadd" in text
    assert "pto.tstore" in text
