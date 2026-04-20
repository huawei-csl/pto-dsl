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

    subtensor_col_src = pto.SubTensorType(shape=[1, tile_cols], dtype=pto_dtype)
    subtensor_row_src = pto.SubTensorType(shape=[tile_rows, 1], dtype=pto_dtype)
    subtensor_scalar = pto.SubTensorType(shape=[1, 1], dtype=pto_dtype)
    subtensor_dst = pto.SubTensorType(shape=[tile_rows, tile_cols], dtype=pto_dtype)

    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[tile_rows, tile_cols],
        valid_shape=[-1, -1],
        dtype=pto_dtype,
        memory_space="VEC",
        config=tile_cfg,
    )

    x_col_tile_cfg = pto.TileBufConfig(blayout="ColMajor")
    x_col_tile_type = pto.TileBufType(
        shape=[tile_rows, 1],
        valid_shape=[-1, 1],
        dtype=pto_dtype,
        memory_space="VEC",
        config=x_col_tile_cfg,
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
        "x_col_tile_type": x_col_tile_type,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
    }


def build_col_expand(dtype="fp32"):
    """Column-wise broadcast: X[i, j] = y[j]"""
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
    """Row-wise broadcast: Y[i, j] = x[i]"""
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
    """Fused col-expand: Z[i,j] = Y[i,j] op x[j]"""
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

            tv_x = pto.as_tensor(
                tensor2d_type,
                ptr=x_ptr,
                shape=[c1, n_cols],
                strides=[n_cols, c1],
            )
            tv_y = pto.as_tensor(
                tensor2d_type,
                ptr=y_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )
            tv_z = pto.as_tensor(
                tensor2d_type,
                ptr=z_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )

            for col in pto.range(col_start, col_end, c_tile_cols):
                cols_this = s.min_u(c_tile_cols, col_end - col)

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
    """Fused row-expand: Z[i,j] = Y[i,j] op x[i]."""
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

            tv_x = pto.as_tensor(
                tensor2d_type,
                ptr=x_ptr,
                shape=[batch, c1],
                strides=[c1, c1],
                layout="DN",
            )
            tv_y = pto.as_tensor(
                tensor2d_type,
                ptr=y_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )
            tv_z = pto.as_tensor(
                tensor2d_type,
                ptr=z_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )

            for row in pto.range(row_start, row_end, c_tile_rows):
                rows_this = s.min_u(c_tile_rows, row_end - row)

                # ColMajor [tile_rows, 1] src1 => fused op uses vbrcb to
                # broadcast x[i] across all output columns internally.
                tb_x = pto.alloc_tile(x_col_tile_type, valid_row=rows_this)
                sv_x = pto.slice_view(
                    subtensor_row_src,
                    source=tv_x,
                    offsets=[row, c0],
                    sizes=[rows_this, c1],
                )
                pto.load(sv_x, tb_x)

                for col in pto.range(c0, n_cols, c_tile_cols):
                    cols_this = s.min_u(c_tile_cols, n_cols - col)

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

                    tb_y = pto.alloc_tile(
                        tile_type, valid_row=rows_this, valid_col=cols_this
                    )
                    pto.load(sv_y, tb_y)

                    tb_dst = pto.alloc_tile(
                        tile_type, valid_row=rows_this, valid_col=cols_this
                    )
                    row_op(tb_y, tb_x, tb_dst)

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=list(builders), default="colexpand")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()

    print(builders[args.mode](dtype=args.dtype))
