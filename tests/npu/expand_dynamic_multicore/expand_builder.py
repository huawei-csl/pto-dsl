from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

_TILE_ROWS = 32
_TILE_COLS = 32


def meta_data_expand(dtype="fp32"):
    pto_dtype = {"fp16": pto.float16, "fp32": pto.float32}[dtype]
    ptr_type = pto.PtrType(pto_dtype)
    index_dtype = pto.int32

    tile_rows = _TILE_ROWS
    tile_cols = _TILE_COLS

    tensor2d_type = pto.TensorType(rank=2, dtype=pto_dtype)

    # For col_expand: src slice is [1, tile_cols] (one row of the input vector)
    subtensor_col_src = pto.SubTensorType(shape=[1, tile_cols], dtype=pto_dtype)
    # For row_expand: src slice is [tile_rows, 1] (one column of the input vector)
    subtensor_row_src = pto.SubTensorType(shape=[tile_rows, 1], dtype=pto_dtype)
    # For loading a single scalar element [1, 1]
    subtensor_scalar = pto.SubTensorType(shape=[1, 1], dtype=pto_dtype)
    # For loading/storing the 2D matrix
    subtensor_dst = pto.SubTensorType(shape=[tile_rows, tile_cols], dtype=pto_dtype)

    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[tile_rows, tile_cols],
        valid_shape=[-1, -1],
        dtype=pto_dtype,
        memory_space="VEC",
        config=tile_cfg,
    )
    return {
        "ptr_type": ptr_type,
        "pto_dtype": pto_dtype,
        "index_dtype": index_dtype,
        "tensor2d_type": tensor2d_type,
        "subtensor_col_src": subtensor_col_src,
        "subtensor_row_src": subtensor_row_src,
        "subtensor_scalar": subtensor_scalar,
        "subtensor_dst": subtensor_dst,
        "tile_type": tile_type,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
    }


def build_col_expand(dtype="fp32"):
    """
    Column-wise broadcast: replicate each element of y[j] across all rows.

    Semantics:
        X[i, j] = y[j]
    """
    _meta_data = lambda: meta_data_expand(dtype=dtype)

    @to_ir_module(meta_data=_meta_data)
    def _kernel(
        src_ptr: "ptr_type",
        dst_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile_rows = const(tile_rows)
        c_tile_cols = const(tile_cols)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            cid = pto.get_block_idx()
            sub_bid = pto.get_subblock_idx()
            sub_bnum = pto.get_subblock_num()
            vid = s.index_cast(cid * sub_bnum + sub_bid)
            num_cores = s.index_cast(pto.get_block_num() * sub_bnum)

            cols_per_core = s.ceil_div(n_cols, num_cores)
            col_start = vid * cols_per_core
            col_end = s.min_u(col_start + cols_per_core, n_cols)

            # src[n_cols] represented as 2D [1, n_cols] for uniform slice_view usage
            tv_src = pto.as_tensor(
                tensor2d_type,
                ptr=src_ptr,
                shape=[c1, n_cols],
                strides=[n_cols, c1],
            )
            tv_dst = pto.as_tensor(
                tensor2d_type,
                ptr=dst_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )

            for col in pto.range(col_start, col_end, c_tile_cols):
                cols_this = s.min_u(c_tile_cols, col_end - col)

                # Load one row of src into the src tile (valid_row=1)
                tb_src = pto.alloc_tile(tile_type, valid_row=c1, valid_col=cols_this)
                sv_src = pto.slice_view(
                    subtensor_col_src,
                    source=tv_src,
                    offsets=[c0, col],
                    sizes=[c1, cols_this],
                )
                pto.load(sv_src, tb_src)

                for row in pto.range(c0, batch, c_tile_rows):
                    rows_this = s.min_u(c_tile_rows, batch - row)

                    tb_dst = pto.alloc_tile(
                        tile_type, valid_row=rows_this, valid_col=cols_this
                    )
                    tile.col_expand(tb_src, tb_dst)

                    sv_dst = pto.slice_view(
                        subtensor_dst,
                        source=tv_dst,
                        offsets=[row, col],
                        sizes=[rows_this, cols_this],
                    )
                    pto.store(tb_dst, sv_dst)

    return _kernel


def build_row_expand(dtype="fp32"):
    """
    Row-wise broadcast: replicate each element of x[i] across all columns.

    Semantics:
        Y[i,j] = x[i]
    """
    _meta_data = lambda: meta_data_expand(dtype=dtype)

    @to_ir_module(meta_data=_meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        y_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile_rows = const(tile_rows)
        c_tile_cols = const(tile_cols)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            cid = pto.get_block_idx()
            sub_bid = pto.get_subblock_idx()
            sub_bnum = pto.get_subblock_num()
            vid = s.index_cast(cid * sub_bnum + sub_bid)
            num_cores = s.index_cast(pto.get_block_num() * sub_bnum)

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = vid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, batch)

            # x[batch] represented as 2D [batch, 1] for uniform slice_view usage
            tv_x = pto.as_tensor(
                tensor2d_type,
                ptr=x_ptr,
                shape=[batch, c1],
                strides=[c1, c1],
            )
            tv_y = pto.as_tensor(
                tensor2d_type,
                ptr=y_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )

            for row in pto.range(row_start, row_end, c_tile_rows):
                rows_this = s.min_u(c_tile_rows, row_end - row)

                # Load one column of x into the src tile (valid_col=1)
                tb_src = pto.alloc_tile(tile_type, valid_row=rows_this, valid_col=c1)
                sv_x = pto.slice_view(
                    subtensor_row_src,
                    source=tv_x,
                    offsets=[row, c0],
                    sizes=[rows_this, c1],
                )
                pto.load(sv_x, tb_src)

                for col in pto.range(c0, n_cols, c_tile_cols):
                    cols_this = s.min_u(c_tile_cols, n_cols - col)

                    tb_dst = pto.alloc_tile(
                        tile_type, valid_row=rows_this, valid_col=cols_this
                    )
                    tile.row_expand(tb_src, tb_dst)

                    sv_y = pto.slice_view(
                        subtensor_dst,
                        source=tv_y,
                        offsets=[row, col],
                        sizes=[rows_this, cols_this],
                    )
                    pto.store(tb_dst, sv_y)

    return _kernel


# Fused col-expand ops: dst[i,j] = src0[i,j] op src1[0,j]
# src1 is a col-vector tile (valid_row=1, valid_col=cols_this)
# so src1[0,j] = x[col+j] per the hardware op convention.
_COL_EXPAND_FUSED_OPS = {
    "colexpand_add": tile.col_expand_add,
    "colexpand_sub": tile.col_expand_sub,
    "colexpand_div": tile.col_expand_div,
    "colexpand_mul": tile.col_expand_mul,
    "colexpand_min": tile.col_expand_min,
    "colexpand_max": tile.col_expand_max,
    "colexpand_expdif": tile.col_expand_expdif,
}


def _build_col_expand_fused(kind, dtype="fp32"):
    """
    Fused col-expand: apply element-wise op between Y[i,j] and x[j].

    Semantics:
        colexpand_mul: Z[i,j] = Y[i,j] * x[j]
        colexpand_sub: Z[i,j] = Y[i,j] - x[j]
        colexpand_div: Z[i,j] = Y[i,j] / x[j]
        colexpand_min: Z[i,j] = min(Y[i,j], x[j])
        colexpand_max: Z[i,j] = max(Y[i,j], x[j])
    """
    col_op = _COL_EXPAND_FUSED_OPS[kind]
    _meta_data = lambda: meta_data_expand(dtype=dtype)

    @to_ir_module(meta_data=_meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        y_ptr: "ptr_type",
        z_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile_rows = const(tile_rows)
        c_tile_cols = const(tile_cols)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            cid = pto.get_block_idx()
            sub_bid = pto.get_subblock_idx()
            sub_bnum = pto.get_subblock_num()
            vid = s.index_cast(cid * sub_bnum + sub_bid)
            num_cores = s.index_cast(pto.get_block_num() * sub_bnum)

            cols_per_core = s.ceil_div(n_cols, num_cores)
            col_start = vid * cols_per_core
            col_end = s.min_u(col_start + cols_per_core, n_cols)

            # x[n_cols] represented as 2D [1, n_cols]
            tv_x = pto.as_tensor(
                tensor2d_type,
                ptr=x_ptr,
                shape=[c1, n_cols],
                strides=[n_cols, c1],
            )
            # y[batch, n_cols] - input matrix (src0)
            tv_y = pto.as_tensor(
                tensor2d_type,
                ptr=y_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )
            # z[batch, n_cols] - output matrix (dst)
            tv_z = pto.as_tensor(
                tensor2d_type,
                ptr=z_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )

            for col in pto.range(col_start, col_end, c_tile_cols):
                cols_this = s.min_u(c_tile_cols, col_end - col)

                # Load x[col:col+cols_this] into a [1, cols_this] tile
                tb_src1 = pto.alloc_tile(tile_type, valid_row=c1, valid_col=cols_this)
                sv_x = pto.slice_view(
                    subtensor_col_src,
                    source=tv_x,
                    offsets=[c0, col],
                    sizes=[c1, cols_this],
                )
                pto.load(sv_x, tb_src1)

                for row in pto.range(c0, batch, c_tile_rows):
                    rows_this = s.min_u(c_tile_rows, batch - row)

                    sv_y = pto.slice_view(
                        subtensor_dst,
                        source=tv_y,
                        offsets=[row, col],
                        sizes=[rows_this, cols_this],
                    )
                    sv_z = pto.slice_view(
                        subtensor_dst,
                        source=tv_z,
                        offsets=[row, col],
                        sizes=[rows_this, cols_this],
                    )

                    tb_src0 = pto.alloc_tile(
                        tile_type, valid_row=rows_this, valid_col=cols_this
                    )
                    pto.load(sv_y, tb_src0)

                    tb_dst = pto.alloc_tile(
                        tile_type, valid_row=rows_this, valid_col=cols_this
                    )
                    col_op(tb_src0, tb_src1, tb_dst)

                    pto.store(tb_dst, sv_z)

    return _kernel


def build_col_expand_sub(dtype="fp32"):
    return _build_col_expand_fused("colexpand_sub", dtype=dtype)


def build_col_expand_div(dtype="fp32"):
    return _build_col_expand_fused("colexpand_div", dtype=dtype)


def build_col_expand_mul(dtype="fp32"):
    return _build_col_expand_fused("colexpand_mul", dtype=dtype)


def build_col_expand_min(dtype="fp32"):
    return _build_col_expand_fused("colexpand_min", dtype=dtype)


def build_col_expand_max(dtype="fp32"):
    return _build_col_expand_fused("colexpand_max", dtype=dtype)


def build_col_expand_add(dtype="fp32"):
    return _build_col_expand_fused("colexpand_add", dtype=dtype)


def build_col_expand_expdif(dtype="fp32"):
    return _build_col_expand_fused("colexpand_expdif", dtype=dtype)


# Fused row-expand ops: dst[i,j] = src0[i,j] op src1[0,i]
# src1 is a row-vector tile (valid_row=1, valid_col=rows_this)
# so src1[0,i] = x[row+i] per the hardware op convention.
_ROW_EXPAND_FUSED_OPS = {
    "rowexpand_add": tile.row_expand_add,
    "rowexpand_mul": tile.row_expand_mul,
    "rowexpand_sub": tile.row_expand_sub,
    "rowexpand_div": tile.row_expand_div,
    "rowexpand_min": tile.row_expand_min,
    "rowexpand_max": tile.row_expand_max,
    "rowexpand_expdif": tile.row_expand_expdif,
}


def _build_row_expand_fused(kind, dtype="fp32"):
    """
    Fused row-expand: apply element-wise op between Y[i,j] and x[i].

    Semantics:
        expand_add: Z[i,j] = Y[i,j] + x[i]
        expand_mul: Z[i,j] = Y[i,j] * x[i]
        expand_sub: Z[i,j] = Y[i,j] - x[i]
        expand_div: Z[i,j] = Y[i,j] / x[i]

    src1 tile is a scalar [1, 1]: src1[0,0] = x[row], one row at a time.
    """
    row_op = _ROW_EXPAND_FUSED_OPS[kind]
    _meta_data = lambda: meta_data_expand(dtype=dtype)

    @to_ir_module(meta_data=_meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        y_ptr: "ptr_type",
        z_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile_cols = const(tile_cols)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            cid = pto.get_block_idx()
            sub_bid = pto.get_subblock_idx()
            sub_bnum = pto.get_subblock_num()
            vid = s.index_cast(cid * sub_bnum + sub_bid)
            num_cores = s.index_cast(pto.get_block_num() * sub_bnum)

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = vid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, batch)

            # y[batch, n_cols] - input matrix (src0)
            tv_y = pto.as_tensor(
                tensor2d_type,
                ptr=y_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )
            # z[batch, n_cols] - output matrix (dst)
            tv_z = pto.as_tensor(
                tensor2d_type,
                ptr=z_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )
            # x as column vector [batch, 1]: x[row] stored at tv_x[row, 0]
            tv_x = pto.as_tensor(
                tensor2d_type,
                ptr=x_ptr,
                shape=[batch, c1],
                strides=[c1, c1],
            )

            # Process one row at a time so tb_src1 always has rows_this=1,
            # making src1[0,0] = x[row] unambiguous for both row/col conventions.
            for row in pto.range(row_start, row_end, c1):
                # Load scalar x[row] into a [1, 1] tile: src1[0,0] = x[row]
                tb_src1 = pto.alloc_tile(tile_type, valid_row=c1, valid_col=c1)
                sv_x = pto.slice_view(
                    subtensor_scalar,
                    source=tv_x,
                    offsets=[row, c0],
                    sizes=[c1, c1],
                )
                pto.load(sv_x, tb_src1)

                for col in pto.range(c0, n_cols, c_tile_cols):
                    cols_this = s.min_u(c_tile_cols, n_cols - col)

                    sv_y = pto.slice_view(
                        subtensor_col_src,
                        source=tv_y,
                        offsets=[row, col],
                        sizes=[c1, cols_this],
                    )
                    sv_z = pto.slice_view(
                        subtensor_col_src,
                        source=tv_z,
                        offsets=[row, col],
                        sizes=[c1, cols_this],
                    )

                    # src0 = one row of Y, src1 = scalar x[row], dst = one row of Z
                    tb_src0 = pto.alloc_tile(
                        tile_type, valid_row=c1, valid_col=cols_this
                    )
                    pto.load(sv_y, tb_src0)

                    tb_dst = pto.alloc_tile(
                        tile_type, valid_row=c1, valid_col=cols_this
                    )
                    row_op(tb_src0, tb_src1, tb_dst)

                    pto.store(tb_dst, sv_z)

    return _kernel


def build_row_expand_add(dtype="fp32"):
    return _build_row_expand_fused("rowexpand_add", dtype=dtype)


def build_row_expand_mul(dtype="fp32"):
    return _build_row_expand_fused("rowexpand_mul", dtype=dtype)


def build_row_expand_sub(dtype="fp32"):
    return _build_row_expand_fused("rowexpand_sub", dtype=dtype)


def build_row_expand_div(dtype="fp32"):
    return _build_row_expand_fused("rowexpand_div", dtype=dtype)


def build_row_expand_min(dtype="fp32"):
    return _build_row_expand_fused("rowexpand_min", dtype=dtype)


def build_row_expand_max(dtype="fp32"):
    return _build_row_expand_fused("rowexpand_max", dtype=dtype)


def build_row_expand_expdif(dtype="fp32"):
    return _build_row_expand_fused("rowexpand_expdif", dtype=dtype)


if __name__ == "__main__":
    import argparse

    _MODES = [
        "colexpand",
        "colexpand_sub",
        "colexpand_div",
        "colexpand_mul",
        "colexpand_min",
        "colexpand_max",
        "colexpand_add",
        "colexpand_expdif",
        "rowexpand",
        "rowexpand_add",
        "rowexpand_mul",
        "rowexpand_sub",
        "rowexpand_div",
        "rowexpand_min",
        "rowexpand_max",
        "rowexpand_expdif",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=_MODES, default="colexpand")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()

    builders = {
        "colexpand": build_col_expand,
        "colexpand_sub": build_col_expand_sub,
        "colexpand_div": build_col_expand_div,
        "colexpand_mul": build_col_expand_mul,
        "colexpand_min": build_col_expand_min,
        "colexpand_max": build_col_expand_max,
        "colexpand_add": build_col_expand_add,
        "colexpand_expdif": build_col_expand_expdif,
        "rowexpand": build_row_expand,
        "rowexpand_add": build_row_expand_add,
        "rowexpand_mul": build_row_expand_mul,
        "rowexpand_sub": build_row_expand_sub,
        "rowexpand_div": build_row_expand_div,
        "rowexpand_min": build_row_expand_min,
        "rowexpand_max": build_row_expand_max,
        "rowexpand_expdif": build_row_expand_expdif,
    }

    print(builders[args.mode](dtype=args.dtype))
