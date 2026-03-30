import re

import pytest

from mlir.dialects import pto as mlir_pto
from mlir.ir import Context, IndexType, Location

from ptodsl import micro, pto, tile, to_ir_module
from ptodsl import scalar as s


const = s.const


class _Box:
    def __init__(self, raw):
        self.raw = raw


def _normalized_pto_lines(text):
    return [
        re.sub(r"%[A-Za-z0-9._]+", "%v", line.strip())
        for line in text.splitlines()
        if "pto." in line
    ]


def test_pythonic_type_factories_match_low_level_builders():
    with Context() as ctx, Location.unknown():
        mlir_pto.register_dialect(ctx, load=True)

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


def _meta_data():
    dtype = pto.float32
    return {
        "ptr_t": pto.ptr(dtype),
        "index_t": IndexType.get(),
        "tensor_t": pto.TensorType(rank=2, dtype=dtype),
        "subtensor_t": pto.SubTensorType(shape=[32, 32], dtype=dtype),
        "tile_t": pto.TileBufType(
            shape=[32, 32],
            valid_shape=[32, 32],
            dtype=dtype,
            memory_space="VEC",
        ),
    }


def _build_pythonic_module():
    def kernel(
        src0: "ptr_t",
        src1: "ptr_t",
        dst: "ptr_t",
        rows: "index_t",
        cols: "index_t",
    ) -> None:
        c0 = const(0)

        tv0 = pto.make_tensor(src0, shape=[rows, cols], dtype=pto.float32)
        tv1 = pto.make_tensor(src1, shape=[rows, cols], dtype=pto.float32)
        tv2 = pto.make_tensor(dst, shape=[rows, cols], dtype=pto.float32)

        sv0 = tv0.slice([c0, c0], [32, 32])
        sv1 = tv1.slice([c0, c0], [32, 32])
        sv2 = tv2.slice([c0, c0], [32, 32])

        tile_spec = pto.make_tile_buffer(pto.float32, [32, 32], space="VEC")
        with pto.vector_section():
            tb0 = tile_spec.alloc()
            tb1 = tile_spec.alloc()
            tb2 = tile_spec.alloc()
            tb0.load_from(sv0)
            tb1.load_from(sv1)
            tile.add(tb0, tb1, tb2)
            tb2.store_to(sv2)

    kernel.__name__ = "vector_add_frontend"
    return to_ir_module(meta_data=_meta_data)(kernel)


def _build_low_level_module():
    def kernel(
        src0: "ptr_t",
        src1: "ptr_t",
        dst: "ptr_t",
        rows: "index_t",
        cols: "index_t",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c32 = const(32)

        tv0 = pto.as_tensor(tensor_t, ptr=src0, shape=[rows, cols], strides=[cols, c1])
        tv1 = pto.as_tensor(tensor_t, ptr=src1, shape=[rows, cols], strides=[cols, c1])
        tv2 = pto.as_tensor(tensor_t, ptr=dst, shape=[rows, cols], strides=[cols, c1])

        sv0 = pto.slice_view(subtensor_t, source=tv0, offsets=[c0, c0], sizes=[c32, c32])
        sv1 = pto.slice_view(subtensor_t, source=tv1, offsets=[c0, c0], sizes=[c32, c32])
        sv2 = pto.slice_view(subtensor_t, source=tv2, offsets=[c0, c0], sizes=[c32, c32])

        with pto.vector_section():
            tb0 = pto.alloc_tile(tile_t)
            tb1 = pto.alloc_tile(tile_t)
            tb2 = pto.alloc_tile(tile_t)
            pto.load(sv0, tb0)
            pto.load(sv1, tb1)
            tile.add(tb0, tb1, tb2)
            pto.store(tb2, sv2)

    kernel.__name__ = "vector_add_frontend"
    return to_ir_module(meta_data=_meta_data)(kernel)


def test_pythonic_tensor_and_tile_flow_matches_low_level_ir():
    pythonic_module = _build_pythonic_module()
    low_level_module = _build_low_level_module()

    pythonic_text = str(pythonic_module)
    low_level_text = str(low_level_module)

    assert _normalized_pto_lines(pythonic_text) == _normalized_pto_lines(low_level_text)
    text = pythonic_text
    assert "pto.make_tensor_view" in text
    assert "pto.partition_view" in text
    assert "strides = [%arg4, %c1]" in text
    assert text.count("pto.alloc_tile") == 3
    assert "pto.tload" in text
    assert "pto.tstore" in text


def test_tensor_view_slice_requires_static_shape_for_dynamic_sizes():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    def kernel(src: "ptr_t", total_elements: "index_t") -> None:
        tv = pto.make_tensor(src, shape=[total_elements], dtype=pto.float32)
        tv.slice([0], [total_elements])

    with pytest.raises(TypeError, match="requires `static_shape=`"):
        to_ir_module(meta_data=meta_data)(kernel)


def test_tensor_view_slice_accepts_dynamic_sizes_with_static_shape():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    def kernel(src: "ptr_t", total_elements: "index_t", tile_extent: "index_t") -> None:
        tv = pto.make_tensor(src, shape=[total_elements], dtype=pto.float32)
        tv.slice([0], [tile_extent], static_shape=[128])

    module = to_ir_module(meta_data=meta_data)(kernel)
    text = str(module)

    assert "pto.partition_view" in text
    assert "!pto.partition_tensor_view<128xf32>" in text


def test_micro_ops_accept_objects_with_raw_values():
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
