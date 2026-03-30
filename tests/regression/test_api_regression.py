from types import SimpleNamespace

from mlir.ir import IndexType

from ptodsl import Constexpr, JitWrapper, const_expr, pto, tile, to_ir_module
from ptodsl import scalar as s


const = s.const


class _FakeType:
    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


def test_low_level_tensor_tile_flow_regression():
    def meta_data():
        dtype = pto.float32
        return {
            "ptr_t": pto.PtrType(dtype),
            "tensor_t": pto.TensorType(rank=2, dtype=dtype),
            "subtensor_t": pto.SubTensorType(shape=[32, 32], dtype=dtype),
            "tile_t": pto.TileBufType(
                shape=[32, 32],
                valid_shape=[32, 32],
                dtype=dtype,
                memory_space="VEC",
            ),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def kernel(src: "ptr_t", dst: "ptr_t", rows: "index_t", cols: "index_t") -> None:
        c0 = const(0)
        c1 = const(1)
        c32 = const(32)

        src_view = pto.as_tensor(tensor_t, ptr=src, shape=[rows, cols], strides=[cols, c1])
        dst_view = pto.as_tensor(tensor_t, ptr=dst, shape=[rows, cols], strides=[cols, c1])
        src_tile = pto.slice_view(subtensor_t, source=src_view, offsets=[c0, c0], sizes=[c32, c32])
        dst_tile = pto.slice_view(subtensor_t, source=dst_view, offsets=[c0, c0], sizes=[c32, c32])

        with pto.vector_section():
            tb = pto.alloc_tile(tile_t)
            pto.load(src_tile, tb)
            tile.add(tb, tb, tb)
            pto.store(tb, dst_tile)

    text = str(kernel)

    assert "pto.make_tensor_view" in text
    assert "pto.partition_view" in text
    assert "pto.tadd" in text
    assert "pto.tstore" in text


def test_pythonic_tensor_tile_flow_regression():
    def meta_data():
        return {
            "ptr_t": pto.ptr(pto.float32),
            "index_t": IndexType.get(),
        }

    @to_ir_module(meta_data=meta_data)
    def kernel(src: "ptr_t", dst: "ptr_t", rows: "index_t", cols: "index_t") -> None:
        src_view = pto.make_tensor(src, shape=[rows, cols], dtype=pto.float32)
        dst_view = pto.make_tensor(dst, shape=[rows, cols], dtype=pto.float32)

        with pto.vector_section():
            buf = pto.make_tile_buffer(pto.float32, [32, 32], space="VEC").alloc()
            buf.load_from(src_view.slice([0, 0], [32, 32]))
            buf.store_to(dst_view.slice([0, 0], [32, 32]))

    text = str(kernel)

    assert "pto.make_tensor_view" in text
    assert "strides = [%arg3, %c1]" in text
    assert "pto.partition_view" in text
    assert "pto.tload" in text
    assert "pto.tstore" in text


def test_mixed_tile_and_micro_regression():
    def meta_data():
        dtype = pto.float32
        return {
            "ptr_t": pto.ptr(dtype, space="VEC"),
            "index_t": IndexType.get(),
            "tile_t": pto.TileBufType(
                shape=[1, 64],
                valid_shape=[1, 64],
                dtype=dtype,
                memory_space="VEC",
            ),
        }

    @to_ir_module(meta_data=meta_data)
    def kernel(src: "ptr_t", dst: "ptr_t", offset: "index_t") -> None:
        with pto.vector_section():
            pto.alloc_tile(tile_t)
            mask = pto.pset_b32(pto.MaskType(), "PAT_ALL")
            vec = pto.vlds(pto.VRegType(64, pto.float32), src, offset)
            pto.vsts(vec, dst, offset, mask)

    text = str(kernel)

    assert "pto.alloc_tile" in text
    assert "pto.pset_b32" in text
    assert "pto.vlds" in text
    assert "pto.vsts" in text


def test_constexpr_specialization_and_jit_caller_regression():
    def meta_data(TILE):
        return {
            "tile_t": pto.TileBufType(
                shape=[1, TILE],
                valid_shape=[1, TILE],
                dtype=pto.float32,
                memory_space="VEC",
            ),
        }

    @to_ir_module(meta_data=meta_data)
    def kernel(TILE: Constexpr[int]) -> None:
        if const_expr(TILE >= 64):
            with pto.vector_section():
                pto.alloc_tile(tile_t)

    module = kernel(TILE=64)
    text = str(module)

    assert "func.func @kernel()" in text
    assert "scf.if" not in text
    assert text.count("pto.alloc_tile") == 1

    wrapper = JitWrapper(kernel.__wrapped__, meta_data=meta_data, block_dim=4)
    wrapper._arg_types = []
    wrapper._runtime_params = []
    caller_cpp = wrapper._generate_caller_cpp("generated.cpp")

    assert "TILE" not in caller_cpp
    assert 'extern "C" void call_kernel(uint32_t blockDim, void *stream)' in caller_cpp
