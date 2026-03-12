from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s


def meta_data():
    src_dtype = pto.float16
    dst_dtype = pto.float32

    ptr_src = pto.PtrType(src_dtype)
    ptr_dst = pto.PtrType(dst_dtype)
    tensor_src = pto.TensorType(rank=2, dtype=src_dtype)
    tensor_dst = pto.TensorType(rank=2, dtype=dst_dtype)
    sub_src = pto.SubTensorType(shape=[1, 32], dtype=src_dtype)
    sub_dst = pto.SubTensorType(shape=[1, 32], dtype=dst_dtype)

    cfg = pto.TileBufConfig()
    src_tile = pto.TileBufType(shape=[1, 32], valid_shape=[1, 32], dtype=src_dtype, memory_space="VEC", config=cfg)
    dst_tile = pto.TileBufType(shape=[1, 32], valid_shape=[1, 32], dtype=dst_dtype, memory_space="VEC", config=cfg)

    return {
        "ptr_src": ptr_src,
        "ptr_dst": ptr_dst,
        "tensor_src": tensor_src,
        "tensor_dst": tensor_dst,
        "sub_src": sub_src,
        "sub_dst": sub_dst,
        "src_tile": src_tile,
        "dst_tile": dst_tile,
    }


def vec_cvt_kernel(src_ptr: "ptr_src", dst_ptr: "ptr_dst") -> None:
    c0 = s.const(0)
    c1 = s.const(1)
    c32 = s.const(32)

    src = pto.as_tensor(tensor_src, ptr=src_ptr, shape=[c1, c32], strides=[c32, c1])
    dst = pto.as_tensor(tensor_dst, ptr=dst_ptr, shape=[c1, c32], strides=[c32, c1])

    with pto.vector_section():
        src_tile_buf = pto.alloc_tile(src_tile)
        dst_tile_buf = pto.alloc_tile(dst_tile)

        pto.load(pto.slice_view(sub_src, source=src, offsets=[c0, c0], sizes=[c1, c32]), src_tile_buf)
        tile.cvt(src_tile_buf, dst_tile_buf)
        pto.store(dst_tile_buf, pto.slice_view(sub_dst, source=dst, offsets=[c0, c0], sizes=[c1, c32]))


def test_tcvt_present_in_ir():
    module = to_ir_module(meta_data=meta_data)(vec_cvt_kernel)
    assert "pto.tcvt" in str(module)


def scatter_meta_data():
    dtype = pto.float16
    idx_dtype = pto.int16

    ptr = pto.PtrType(dtype)
    ptr_idx = pto.PtrType(idx_dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    tensor_idx = pto.TensorType(rank=2, dtype=idx_dtype)
    sub_src = pto.SubTensorType(shape=[1, 16], dtype=dtype)
    sub_idx = pto.SubTensorType(shape=[1, 16], dtype=idx_dtype)
    sub_dst = pto.SubTensorType(shape=[4, 16], dtype=dtype)

    cfg = pto.TileBufConfig()
    src_tile = pto.TileBufType(shape=[1, 16], valid_shape=[1, 16], dtype=dtype, memory_space="VEC", config=cfg)
    idx_tile = pto.TileBufType(shape=[1, 16], valid_shape=[1, 16], dtype=idx_dtype, memory_space="VEC", config=cfg)
    dst_tile = pto.TileBufType(shape=[4, 16], valid_shape=[4, 16], dtype=dtype, memory_space="VEC", config=cfg)

    return {
        "ptr": ptr,
        "ptr_idx": ptr_idx,
        "tensor": tensor,
        "tensor_idx": tensor_idx,
        "sub_src": sub_src,
        "sub_idx": sub_idx,
        "sub_dst": sub_dst,
        "src_tile": src_tile,
        "idx_tile": idx_tile,
        "dst_tile": dst_tile,
    }


def vec_scatter_kernel(src_ptr: "ptr", idx_ptr: "ptr_idx", dst_ptr: "ptr") -> None:
    c0 = s.const(0)
    c1 = s.const(1)
    c4 = s.const(4)
    c16 = s.const(16)

    src = pto.as_tensor(tensor, ptr=src_ptr, shape=[c1, c16], strides=[c16, c1])
    idx = pto.as_tensor(tensor_idx, ptr=idx_ptr, shape=[c1, c16], strides=[c16, c1])
    dst = pto.as_tensor(tensor, ptr=dst_ptr, shape=[c4, c16], strides=[c16, c1])

    with pto.vector_section():
        src_tile_buf = pto.alloc_tile(src_tile)
        idx_tile_buf = pto.alloc_tile(idx_tile)
        dst_tile_buf = pto.alloc_tile(dst_tile)

        pto.load(pto.slice_view(sub_dst, source=dst, offsets=[c0, c0], sizes=[c4, c16]), dst_tile_buf)
        pto.load(pto.slice_view(sub_src, source=src, offsets=[c0, c0], sizes=[c1, c16]), src_tile_buf)
        pto.load(pto.slice_view(sub_idx, source=idx, offsets=[c0, c0], sizes=[c1, c16]), idx_tile_buf)
        tile.scatter(src_tile_buf, idx_tile_buf, dst_tile_buf)
        pto.store(dst_tile_buf, pto.slice_view(sub_dst, source=dst, offsets=[c0, c0], sizes=[c4, c16]))


def test_tscatter_present_in_ir():
    module = to_ir_module(meta_data=scatter_meta_data)(vec_scatter_kernel)
    assert "pto.tscatter" in str(module)


def scalar_tile_meta_data():
    dtype = pto.float32
    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    sub = pto.SubTensorType(shape=[1, 32], dtype=dtype)
    cfg = pto.TileBufConfig()
    tile_buf = pto.TileBufType(shape=[1, 32], valid_shape=[1, 32], dtype=dtype, memory_space="VEC", config=cfg)
    return {
        "ptr": ptr,
        "tensor": tensor,
        "sub": sub,
        "tile_buf": tile_buf,
    }


def scalar_tile_kernel(src_ptr: "ptr", dst_ptr: "ptr") -> None:
    c0 = s.const(0)
    c1 = s.const(1)
    c32 = s.const(32)
    c15 = s.const(1.5, dtype=pto.float32)
    c05 = s.const(0.5, dtype=pto.float32)

    src = pto.as_tensor(tensor, ptr=src_ptr, shape=[c1, c32], strides=[c32, c1])
    dst = pto.as_tensor(tensor, ptr=dst_ptr, shape=[c1, c32], strides=[c32, c1])

    with pto.vector_section():
        src_tile = pto.alloc_tile(tile_buf)
        const_tile = pto.alloc_tile(tile_buf)
        out_tile = pto.alloc_tile(tile_buf)
        tmp_tile = pto.alloc_tile(tile_buf)

        pto.load(pto.slice_view(sub, source=src, offsets=[c0, c0], sizes=[c1, c32]), src_tile)
        tile.expands(c15, const_tile)
        tile.muls(src_tile, c05, tmp_tile)
        tile.sub(const_tile, tmp_tile, out_tile)
        pto.store(out_tile, pto.slice_view(sub, source=dst, offsets=[c0, c0], sizes=[c1, c32]))


def test_scalar_tile_helpers_present_in_ir():
    module = to_ir_module(meta_data=scalar_tile_meta_data)(scalar_tile_kernel)
    ir = str(module)
    assert "pto.texpands" in ir
    assert "pto.tmuls" in ir
