from ptodsl import pto, to_ir_module


def meta_data():
    src_dtype = pto.float16
    dst_dtype = pto.float32

    ptr_src = pto.PtrType(src_dtype)
    ptr_dst = pto.PtrType(dst_dtype)
    tensor_src = pto.TensorType(rank=2, dtype=src_dtype)
    tensor_dst = pto.TensorType(rank=2, dtype=dst_dtype)
    sub_src = pto.SubTensorType(shape=[1, 32], dtype=src_dtype)
    sub_dst = pto.SubTensorType(shape=[1, 32], dtype=dst_dtype)

    cfg = pto.TileConfig()
    src_tile = pto.TileType(shape=[1, 32], valid_shape=[1, 32], dtype=src_dtype, memory_space="VEC", config=cfg)
    dst_tile = pto.TileType(shape=[1, 32], valid_shape=[1, 32], dtype=dst_dtype, memory_space="VEC", config=cfg)

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
    c0 = pto.const(0)
    c1 = pto.const(1)
    c32 = pto.const(32)

    src = pto.as_tensor(tensor_src, ptr=src_ptr, shape=[c1, c32], strides=[c32, c1])
    dst = pto.as_tensor(tensor_dst, ptr=dst_ptr, shape=[c1, c32], strides=[c32, c1])

    with pto.vector_section():
        src_tile_buf = pto.alloc_tile(src_tile)
        dst_tile_buf = pto.alloc_tile(dst_tile)

        pto.load(
            pto.slice_view(sub_src, source=src, offsets=[c0, c0], sizes=[c1, c32]),
            src_tile_buf,
        )
        pto.cvt(src_tile_buf, dst_tile_buf)
        pto.store(
            dst_tile_buf,
            pto.slice_view(sub_dst, source=dst, offsets=[c0, c0], sizes=[c1, c32]),
        )


def test_tcvt_present_in_ir():
    module = to_ir_module(meta_data=meta_data)(vec_cvt_kernel)
    assert "pto.tcvt" in str(module)


def store_fp_meta_data():
    src_dtype = pto.float16
    fp_dtype = pto.float32
    dst_dtype = pto.int8

    ptr_src = pto.PtrType(src_dtype)
    ptr_fp = pto.PtrType(fp_dtype)
    ptr_dst = pto.PtrType(dst_dtype)
    tensor_src = pto.TensorType(rank=2, dtype=src_dtype)
    tensor_fp = pto.TensorType(rank=2, dtype=fp_dtype)
    tensor_dst = pto.TensorType(rank=2, dtype=dst_dtype)
    sub_src = pto.SubTensorType(shape=[1, 32], dtype=src_dtype)
    sub_fp = pto.SubTensorType(shape=[1, 32], dtype=fp_dtype)
    sub_dst = pto.SubTensorType(shape=[1, 32], dtype=dst_dtype)

    cfg = pto.TileConfig()
    src_tile = pto.TileType(shape=[1, 32], valid_shape=[1, 32], dtype=src_dtype, memory_space="VEC", config=cfg)
    fp_tile = pto.TileType(shape=[1, 32], valid_shape=[1, 32], dtype=fp_dtype, memory_space="VEC", config=cfg)

    return {
        "ptr_src": ptr_src,
        "ptr_fp": ptr_fp,
        "ptr_dst": ptr_dst,
        "tensor_src": tensor_src,
        "tensor_fp": tensor_fp,
        "tensor_dst": tensor_dst,
        "sub_src": sub_src,
        "sub_fp": sub_fp,
        "sub_dst": sub_dst,
        "src_tile": src_tile,
        "fp_tile": fp_tile,
    }


def store_fp_kernel(src_ptr: "ptr_src", fp_ptr: "ptr_fp", dst_ptr: "ptr_dst") -> None:
    c0 = pto.const(0)
    c1 = pto.const(1)
    c32 = pto.const(32)

    src = pto.as_tensor(tensor_src, ptr=src_ptr, shape=[c1, c32], strides=[c32, c1])
    fp = pto.as_tensor(tensor_fp, ptr=fp_ptr, shape=[c1, c32], strides=[c32, c1])
    dst = pto.as_tensor(tensor_dst, ptr=dst_ptr, shape=[c1, c32], strides=[c32, c1])

    with pto.vector_section():
        src_tile_buf = pto.alloc_tile(src_tile)
        fp_tile_buf = pto.alloc_tile(fp_tile)
        pto.load(pto.slice_view(sub_src, source=src, offsets=[c0, c0], sizes=[c1, c32]), src_tile_buf)
        pto.load(pto.slice_view(sub_fp, source=fp, offsets=[c0, c0], sizes=[c1, c32]), fp_tile_buf)
        pto.store_fp(src_tile_buf, fp_tile_buf, pto.slice_view(sub_dst, source=dst, offsets=[c0, c0], sizes=[c1, c32]))


def test_tstore_fp_present_in_ir():
    module = to_ir_module(meta_data=store_fp_meta_data)(store_fp_kernel)
    assert "pto.tstore_fp" in str(module)


def gemv_meta_data():
    dtype = pto.float16
    acc_dtype = pto.float32
    cfg = pto.TileConfig()
    lhs_tile = pto.TileType(shape=[16, 16], dtype=dtype, memory_space="LEFT", config=cfg)
    rhs_tile = pto.TileType(shape=[1, 16], valid_shape=[1, 16], dtype=dtype, memory_space="RIGHT", config=cfg)
    dst_tile = pto.TileType(shape=[1, 16], valid_shape=[1, 16], dtype=acc_dtype, memory_space="ACC", config=cfg)
    return {
        "lhs_tile": lhs_tile,
        "rhs_tile": rhs_tile,
        "dst_tile": dst_tile,
    }


def gemv_kernel() -> None:
    with pto.cube_section():
        lhs = pto.alloc_tile(lhs_tile)
        rhs = pto.alloc_tile(rhs_tile)
        dst = pto.alloc_tile(dst_tile)
        pto.gemv(lhs, rhs, dst)


def test_tgemv_present_in_ir():
    module = to_ir_module(meta_data=gemv_meta_data)(gemv_kernel)
    assert "pto.tgemv" in str(module)


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

    cfg = pto.TileConfig()
    src_tile = pto.TileType(shape=[1, 16], valid_shape=[1, 16], dtype=dtype, memory_space="VEC", config=cfg)
    idx_tile = pto.TileType(shape=[1, 16], valid_shape=[1, 16], dtype=idx_dtype, memory_space="VEC", config=cfg)
    dst_tile = pto.TileType(shape=[4, 16], valid_shape=[4, 16], dtype=dtype, memory_space="VEC", config=cfg)

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
    c0 = pto.const(0)
    c1 = pto.const(1)
    c4 = pto.const(4)
    c16 = pto.const(16)

    src = pto.as_tensor(tensor, ptr=src_ptr, shape=[c1, c16], strides=[c16, c1])
    idx = pto.as_tensor(tensor_idx, ptr=idx_ptr, shape=[c1, c16], strides=[c16, c1])
    dst = pto.as_tensor(tensor, ptr=dst_ptr, shape=[c4, c16], strides=[c16, c1])

    with pto.vector_section():
        src_tile_buf = pto.alloc_tile(src_tile)
        idx_tile_buf = pto.alloc_tile(idx_tile)
        dst_tile_buf = pto.alloc_tile(dst_tile)

        pto.load(
            pto.slice_view(sub_dst, source=dst, offsets=[c0, c0], sizes=[c4, c16]),
            dst_tile_buf,
        )
        pto.load(
            pto.slice_view(sub_src, source=src, offsets=[c0, c0], sizes=[c1, c16]),
            src_tile_buf,
        )
        pto.load(
            pto.slice_view(sub_idx, source=idx, offsets=[c0, c0], sizes=[c1, c16]),
            idx_tile_buf,
        )
        pto.scatter(src_tile_buf, idx_tile_buf, dst_tile_buf)
        pto.store(
            dst_tile_buf,
            pto.slice_view(sub_dst, source=dst, offsets=[c0, c0], sizes=[c4, c16]),
        )


def test_tscatter_present_in_ir():
    module = to_ir_module(meta_data=scatter_meta_data)(vec_scatter_kernel)
    assert "pto.tscatter" in str(module)


def row_expand_bin_meta_data():
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    sub_full = pto.SubTensorType(shape=[1, 32], dtype=dtype)
    sub_scalar = pto.SubTensorType(shape=[1, 1], dtype=dtype)
    cfg = pto.TileConfig()
    full_tile = pto.TileType(shape=[1, 32], valid_shape=[1, 32], dtype=dtype, memory_space="VEC", config=cfg)
    scalar_tile = pto.TileType(shape=[1, 32], valid_shape=[1, 1], dtype=dtype, memory_space="VEC", config=cfg)
    return {
        "ptr": ptr,
        "tensor": tensor,
        "sub_full": sub_full,
        "sub_scalar": sub_scalar,
        "full_tile": full_tile,
        "scalar_tile": scalar_tile,
    }


def row_expand_bin_kernel(src_ptr: "ptr", scalar_ptr: "ptr", dst_ptr: "ptr") -> None:
    c0 = pto.const(0)
    c1 = pto.const(1)
    c32 = pto.const(32)

    src = pto.as_tensor(tensor, ptr=src_ptr, shape=[c1, c32], strides=[c32, c1])
    scalar = pto.as_tensor(tensor, ptr=scalar_ptr, shape=[c1, c1], strides=[c1, c1])
    dst = pto.as_tensor(tensor, ptr=dst_ptr, shape=[c1, c32], strides=[c32, c1])

    with pto.vector_section():
        src_tile_buf = pto.alloc_tile(full_tile)
        scalar_tile_buf = pto.alloc_tile(scalar_tile)
        dst_tile_buf = pto.alloc_tile(full_tile)

        pto.load(
            pto.slice_view(sub_full, source=src, offsets=[c0, c0], sizes=[c1, c32]),
            src_tile_buf,
        )
        pto.load(
            pto.slice_view(sub_scalar, source=scalar, offsets=[c0, c0], sizes=[c1, c1]),
            scalar_tile_buf,
        )
        pto.row_expand_sub(src_tile_buf, scalar_tile_buf, dst_tile_buf)
        pto.row_expand_div(dst_tile_buf, scalar_tile_buf, dst_tile_buf)
        pto.row_expand_mul(dst_tile_buf, scalar_tile_buf, dst_tile_buf)
        pto.store(
            dst_tile_buf,
            pto.slice_view(sub_full, source=dst, offsets=[c0, c0], sizes=[c1, c32]),
        )


def test_row_expand_bin_ops_present_in_ir():
    module = to_ir_module(meta_data=row_expand_bin_meta_data)(row_expand_bin_kernel)
    text = str(module)
    assert "pto.trowexpandsub" in text
    assert "pto.trowexpanddiv" in text
    assert "pto.trowexpandmul" in text


def col_expand_bin_kernel(src_ptr: "ptr", scalar_ptr: "ptr", dst_ptr: "ptr") -> None:
    c0 = pto.const(0)
    c1 = pto.const(1)
    c32 = pto.const(32)

    src = pto.as_tensor(tensor, ptr=src_ptr, shape=[c32, c1], strides=[c1, c1])
    scalar = pto.as_tensor(tensor, ptr=scalar_ptr, shape=[c1, c1], strides=[c1, c1])
    dst = pto.as_tensor(tensor, ptr=dst_ptr, shape=[c32, c1], strides=[c1, c1])

    with pto.vector_section():
        src_tile_buf = pto.alloc_tile(full_tile)
        scalar_tile_buf = pto.alloc_tile(scalar_tile)
        dst_tile_buf = pto.alloc_tile(full_tile)

        pto.load(
            pto.slice_view(sub_full, source=src, offsets=[c0, c0], sizes=[c32, c1]),
            src_tile_buf,
        )
        pto.load(
            pto.slice_view(sub_scalar, source=scalar, offsets=[c0, c0], sizes=[c1, c1]),
            scalar_tile_buf,
        )
        pto.col_expand_sub(src_tile_buf, scalar_tile_buf, dst_tile_buf)
        pto.col_expand_div(dst_tile_buf, scalar_tile_buf, dst_tile_buf)
        pto.col_expand_mul(dst_tile_buf, scalar_tile_buf, dst_tile_buf)
        pto.store(
            dst_tile_buf,
            pto.slice_view(sub_full, source=dst, offsets=[c0, c0], sizes=[c32, c1]),
        )


def test_col_expand_bin_ops_present_in_ir():
    module = to_ir_module(meta_data=row_expand_bin_meta_data)(col_expand_bin_kernel)
    text = str(module)
    assert "pto.tcolexpandsub" in text
    assert "pto.tcolexpanddiv" in text
    assert "pto.tcolexpandmul" in text


def scalar_ptr_meta_data():
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    return {
        "ptr": ptr,
    }


def scalar_ptr_kernel(src_ptr: "ptr", dst_ptr: "ptr") -> None:
    c0 = pto.const(0)
    val = pto.load_scalar(pto.float16, src_ptr, c0)
    pto.store_scalar(dst_ptr, c0, val)


def test_scalar_ptr_ops_present_in_ir():
    module = to_ir_module(meta_data=scalar_ptr_meta_data)(scalar_ptr_kernel)
    text = str(module)
    assert "pto.load_scalar" in text
    assert "pto.store_scalar" in text


def scalar_tile_meta_data():
    dtype = pto.float32
    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    sub = pto.SubTensorType(shape=[1, 32], dtype=dtype)
    cfg = pto.TileConfig()
    tile_buf = pto.TileType(shape=[1, 32], valid_shape=[1, 32], dtype=dtype, memory_space="VEC", config=cfg)
    return {
        "ptr": ptr,
        "tensor": tensor,
        "sub": sub,
        "tile_buf": tile_buf,
    }


def scalar_tile_kernel(src_ptr: "ptr", dst_ptr: "ptr") -> None:
    c0 = pto.const(0)
    c1 = pto.const(1)
    c32 = pto.const(32)
    c15 = pto.const(1.5, dtype=pto.float32)
    c05 = pto.const(0.5, dtype=pto.float32)

    src = pto.as_tensor(tensor, ptr=src_ptr, shape=[c1, c32], strides=[c32, c1])
    dst = pto.as_tensor(tensor, ptr=dst_ptr, shape=[c1, c32], strides=[c32, c1])

    with pto.vector_section():
        src_tile = pto.alloc_tile(tile_buf)
        const_tile = pto.alloc_tile(tile_buf)
        out_tile = pto.alloc_tile(tile_buf)
        tmp_tile = pto.alloc_tile(tile_buf)

        pto.load(
            pto.slice_view(sub, source=src, offsets=[c0, c0], sizes=[c1, c32]),
            src_tile,
        )
        pto.expands(c15, const_tile)
        pto.muls(src_tile, c05, tmp_tile)
        pto.sub(const_tile, tmp_tile, out_tile)
        pto.store(
            out_tile,
            pto.slice_view(sub, source=dst, offsets=[c0, c0], sizes=[c1, c32]),
        )


def test_scalar_tile_helpers_present_in_ir():
    module = to_ir_module(meta_data=scalar_tile_meta_data)(scalar_tile_kernel)
    ir = str(module)
    assert "pto.texpands" in ir
    assert "pto.tmuls" in ir
