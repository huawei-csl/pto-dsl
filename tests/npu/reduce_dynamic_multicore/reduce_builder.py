from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

# 32 KB of UB
_TILE_SIZE_BYTES = 32 * 1024
_DTYPE_BYTES = {"fp16": 2, "fp32": 4}


def meta_data_row(dtype="fp32"):
    pto_dtype = {"fp16": pto.float16, "fp32": pto.float32}[dtype]
    elements_per_tile = _TILE_SIZE_BYTES // _DTYPE_BYTES[dtype]
    ptr_type = pto.PtrType(pto_dtype)
    index_dtype = pto.int32

    tensor_type = pto.TensorType(rank=1, dtype=pto_dtype)
    subtensor_in = pto.SubTensorType(shape=[1, elements_per_tile], dtype=pto_dtype)

    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[1, elements_per_tile],
        valid_shape=[1, -1],
        dtype=pto_dtype,
        memory_space="VEC",
        config=tile_cfg,
    )

    return {
        "ptr_type": ptr_type,
        "pto_dtype": pto_dtype,
        "elements_per_tile": elements_per_tile,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_in": subtensor_in,
        "tile_type": tile_type,
    }


def meta_data_col(dtype="fp32"):
    pto_dtype = {"fp16": pto.float16, "fp32": pto.float32}[dtype]
    ptr_type = pto.PtrType(pto_dtype)
    index_dtype = pto.int32

    tile_rows = 32
    tile_cols = 32

    tensor2d_type = pto.TensorType(rank=2, dtype=pto_dtype)
    subtensor_in = pto.SubTensorType(shape=[tile_rows, tile_cols], dtype=pto_dtype)
    subtensor_out = pto.SubTensorType(shape=[1, tile_cols], dtype=pto_dtype)

    tile_cfg = pto.TileBufConfig()

    tile_type = pto.TileBufType(
        shape=[tile_rows, tile_cols],
        valid_shape=[-1, -1],
        dtype=pto_dtype,
        memory_space="VEC",
        config=tile_cfg,
    )

    tile_out_type = pto.TileBufType(
        shape=[1, tile_cols],
        valid_shape=[1, -1],
        dtype=pto_dtype,
        memory_space="VEC",
        config=tile_cfg,
    )

    return {
        "ptr_type": ptr_type,
        "pto_dtype": pto_dtype,
        "index_dtype": index_dtype,
        "tensor2d_type": tensor2d_type,
        "subtensor_in": subtensor_in,
        "subtensor_out": subtensor_out,
        "tile_type": tile_type,
        "tile_out_type": tile_out_type,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
    }


_ROW_REDUCE_OPS = {
    "sum": tile.row_sum,
    "min": tile.row_min,
    "max": tile.row_max,
    "prod": tile.row_prod,
}

_COL_REDUCE_OPS = {
    "sum": tile.col_sum,
    "min": lambda src, tmp, dst: tile.col_min(src, dst),
    "max": lambda src, tmp, dst: tile.col_max(src, dst),
    "prod": tile.col_prod,
}

_COL_COMBINE_OPS = {
    "sum": tile.add,
    "min": tile.min,
    "max": tile.max,
    "prod": tile.mul,
}


def build_row_reduce(kind="sum", dtype="fp32"):
    """
    Generic row-wise reduction across columns.

    Semantics:
        y[row] = reduce_j x[row, j]
    """
    if kind not in _ROW_REDUCE_OPS:
        raise ValueError(f"Unsupported row reduction kind: {kind}")

    row_reduce = _ROW_REDUCE_OPS[kind]
    _meta_data = lambda: meta_data_row(dtype=dtype)

    @to_ir_module(meta_data=_meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        y_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            num_cores = s.index_cast(pto.get_block_num())

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, batch)
            num_rows = row_end - row_start

            total_elems = batch * n_cols
            tv_x = pto.as_tensor(
                tensor_type, ptr=x_ptr, shape=[total_elems], strides=[c1]
            )
            tv_y = pto.as_tensor(tensor_type, ptr=y_ptr, shape=[batch], strides=[c1])

            with pto.if_context(num_rows > c0):
                tb_x = pto.alloc_tile(tile_type, valid_col=n_cols)
                tb_out = pto.alloc_tile(tile_type, valid_col=c1)
                tb_tmp = pto.alloc_tile(tile_type, valid_col=n_cols)

                for r in pto.range(c0, num_rows, c1):
                    gm_offset = (row_start + r) * n_cols

                    sv_x = pto.slice_view(
                        subtensor_in,
                        source=tv_x,
                        offsets=[gm_offset],
                        sizes=[n_cols],
                    )

                    sv_y = pto.slice_view(
                        subtensor_in,
                        source=tv_y,
                        offsets=[row_start + r],
                        sizes=[c1],
                    )

                    pto.load(sv_x, tb_x)
                    row_reduce(tb_x, tb_tmp, tb_out)
                    pto.store(tb_out, sv_y)

    return _kernel


def build_col_reduce(kind="sum", dtype="fp32"):
    """
    Generic column-wise reduction across rows.

    Semantics:
        y[col] = reduce_i x[i, col]
    """
    if kind not in _COL_REDUCE_OPS:
        raise ValueError(f"Unsupported column reduction kind: {kind}")

    col_reduce = _COL_REDUCE_OPS[kind]
    combine = _COL_COMBINE_OPS[kind]
    _meta_data = lambda: meta_data_col(dtype=dtype)

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

            cols_per_core = s.ceil_div(n_cols, num_cores)
            col_start = bid * cols_per_core
            col_end = s.min_u(col_start + cols_per_core, n_cols)
            num_cols = col_end - col_start

            tv_x = pto.as_tensor(
                tensor2d_type,
                ptr=x_ptr,
                shape=[batch, n_cols],
                strides=[n_cols, c1],
            )
            tv_y = pto.as_tensor(
                tensor2d_type,
                ptr=y_ptr,
                shape=[c1, n_cols],
                strides=[n_cols, c1],
            )
            for col in pto.range(col_start, col_end, c_tile_cols):
                cols_this = s.min_u(c_tile_cols, col_end - col)
                rows_this0 = s.min_u(c_tile_rows, batch)

                tb_x0 = pto.alloc_tile(
                    tile_type,
                    valid_row=rows_this0,
                    valid_col=cols_this,
                )
                tb_tmp0 = pto.alloc_tile(
                    tile_type,
                    valid_row=rows_this0,
                    valid_col=cols_this,
                )
                if kind in {"min", "max"}:
                    tb_acc = pto.alloc_tile(
                        tile_type, valid_row=c1, valid_col=cols_this
                    )
                else:
                    tb_acc = pto.alloc_tile(tile_out_type, valid_col=cols_this)

                sv_x0 = pto.slice_view(
                    subtensor_in,
                    source=tv_x,
                    offsets=[c0, col],
                    sizes=[rows_this0, cols_this],
                )
                pto.load(sv_x0, tb_x0)
                col_reduce(tb_x0, tb_tmp0, tb_acc)

                for row in pto.range(c_tile_rows, batch, c_tile_rows):
                    rows_this = s.min_u(c_tile_rows, batch - row)

                    tb_x = pto.alloc_tile(
                        tile_type,
                        valid_row=rows_this,
                        valid_col=cols_this,
                    )
                    tb_tmp = pto.alloc_tile(
                        tile_type,
                        valid_row=rows_this,
                        valid_col=cols_this,
                    )
                    if kind in {"min", "max"}:
                        tb_part = pto.alloc_tile(
                            tile_type, valid_row=c1, valid_col=cols_this
                        )
                    else:
                        tb_part = pto.alloc_tile(tile_out_type, valid_col=cols_this)

                    sv_x = pto.slice_view(
                        subtensor_in,
                        source=tv_x,
                        offsets=[row, col],
                        sizes=[rows_this, cols_this],
                    )
                    pto.load(sv_x, tb_x)
                    col_reduce(tb_x, tb_tmp, tb_part)
                    combine(tb_acc, tb_part, tb_acc)

                sv_y = pto.slice_view(
                    subtensor_out,
                    source=tv_y,
                    offsets=[c0, col],
                    sizes=[c1, cols_this],
                )
                pto.store(tb_acc, sv_y)

    return _kernel


def build_rowsum(dtype="fp32"):
    return build_row_reduce("sum", dtype=dtype)


def build_rowmin(dtype="fp32"):
    return build_row_reduce("min", dtype=dtype)


def build_rowmax(dtype="fp32"):
    return build_row_reduce("max", dtype=dtype)


def build_rowprod(dtype="fp32"):
    return build_row_reduce("prod", dtype=dtype)


def build_colsum(dtype="fp32"):
    return build_col_reduce("sum", dtype=dtype)


def build_colmin(dtype="fp32"):
    return build_col_reduce("min", dtype=dtype)


def build_colmax(dtype="fp32"):
    return build_col_reduce("max", dtype=dtype)


def build_colprod(dtype="fp32"):
    return build_col_reduce("prod", dtype=dtype)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "rowsum",
            "rowmin",
            "rowmax",
            "rowprod",
            "colsum",
            "colmin",
            "colmax",
            "colprod",
        ],
        default="rowsum",
    )
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()

    builders = {
        "rowsum": build_rowsum,
        "rowmin": build_rowmin,
        "rowmax": build_rowmax,
        "rowprod": build_rowprod,
        "colsum": build_colsum,
        "colmin": build_colmin,
        "colmax": build_colmax,
        "colprod": build_colprod,
    }

    print(builders[args.mode](dtype=args.dtype))
