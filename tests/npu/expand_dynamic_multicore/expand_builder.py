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
        y_ptr: "ptr_type",
        x_ptr: "ptr_type",
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
            bid = s.index_cast(pto.get_block_idx())
            num_cores = s.index_cast(pto.get_block_num())

            cols_per_core = s.ceil_div(n_cols, num_cores)
            col_start = bid * cols_per_core
            col_end = s.min_u(col_start + cols_per_core, n_cols)

            # y[n_cols] represented as 2D [1, n_cols] for uniform slice_view usage
            tv_y = pto.as_tensor(
                tensor2d_type,
                ptr=y_ptr,
                shape=[c1, n_cols],
                strides=[n_cols, c1],
            )
            tv_x = pto.as_tensor(
                tensor2d_type,
                ptr=x_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )

            for col in pto.range(col_start, col_end, c_tile_cols):
                cols_this = s.min_u(c_tile_cols, col_end - col)

                # Load one row of y into the src tile (valid_row=1)
                tb_src = pto.alloc_tile(tile_type, valid_row=c1, valid_col=cols_this)
                sv_y = pto.slice_view(
                    subtensor_col_src,
                    source=tv_y,
                    offsets=[c0, col],
                    sizes=[c1, cols_this],
                )
                pto.load(sv_y, tb_src)

                for row in pto.range(c0, batch, c_tile_rows):
                    rows_this = s.min_u(c_tile_rows, batch - row)

                    tb_dst = pto.alloc_tile(
                        tile_type, valid_row=rows_this, valid_col=cols_this
                    )
                    tile.col_expand(tb_src, tb_dst)

                    sv_x = pto.slice_view(
                        subtensor_dst,
                        source=tv_x,
                        offsets=[row, col],
                        sizes=[rows_this, cols_this],
                    )
                    pto.store(tb_dst, sv_x)

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
            bid = s.index_cast(pto.get_block_idx())
            num_cores = s.index_cast(pto.get_block_num())

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = bid * rows_per_core
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


# Fused row-expand ops: dst[i,j] = src0[i,j] op src1[0,i]
# src1 is a row-vector tile (valid_row=1, valid_col=rows_this)
# so src1[0,i] = x[row+i] per the hardware op convention.
_ROW_EXPAND_FUSED_OPS = {
    "expand_mul": tile.row_expand_mul,
    "expand_sub": tile.row_expand_sub,
    "expand_div": tile.row_expand_div,
}


def _build_row_expand_fused(kind, dtype="fp32"):
    """
    Fused row-expand: apply element-wise op between Y[i,j] and x[i].

    Semantics:
        expand_mul: Y[i,j] *= x[i]
        expand_sub: Y[i,j] -= x[i]
        expand_div: Y[i,j] /= x[i]

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
            bid = s.index_cast(pto.get_block_idx())
            num_cores = s.index_cast(pto.get_block_num())

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = bid * rows_per_core
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
                    subtensor_row_src,
                    source=tv_x,
                    offsets=[row, c0],
                    sizes=[c1, c1],
                )
                pto.load(sv_x, tb_src1)

                for col in pto.range(c0, n_cols, c_tile_cols):
                    cols_this = s.min_u(c_tile_cols, n_cols - col)

                    sv_y = pto.slice_view(
                        subtensor_dst,
                        source=tv_y,
                        offsets=[row, col],
                        sizes=[c1, cols_this],
                    )
                    sv_z = pto.slice_view(
                        subtensor_dst,
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


def build_row_expand_mul(dtype="fp32"):
    return _build_row_expand_fused("expand_mul", dtype=dtype)


def build_row_expand_sub(dtype="fp32"):
    return _build_row_expand_fused("expand_sub", dtype=dtype)


def build_row_expand_div(dtype="fp32"):
    return _build_row_expand_fused("expand_div", dtype=dtype)


if __name__ == "__main__":
    import argparse

    _MODES = [
        "colexpand",
        "rowexpand",
        "rowexpand_mul",
        "rowexpand_sub",
        "rowexpand_div",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=_MODES, default="colexpand")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()

    builders = {
        "colexpand": build_col_expand,
        "rowexpand": build_row_expand,
        "rowexpand_mul": build_row_expand_mul,
        "rowexpand_sub": build_row_expand_sub,
        "rowexpand_div": build_row_expand_div,
    }

    print(builders[args.mode](dtype=args.dtype))
