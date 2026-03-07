from ptodsl import pto, to_ir_module
from ptodsl import scalar as s

const = s.const

# 32 KB of UB
_TILE_SIZE_BYTES = 32 * 1024
_DTYPE_BYTES = {"float32": 4, "float16": 2}


def meta_data(dtype="float32"):
    pto_dtype = {"float32": pto.float32, "float16": pto.float16}[dtype]
    elements_per_tile = _TILE_SIZE_BYTES // _DTYPE_BYTES[dtype]
    ptr_type = pto.PtrType(pto_dtype)
    index_dtype = pto.int32

    tensor_type = pto.TensorType(rank=1, dtype=pto_dtype)
    subtensor_type = pto.SubTensorType(shape=[1, elements_per_tile], dtype=pto_dtype)

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
        "subtensor_type": subtensor_type,
        "tile_type": tile_type,
    }


def build_unary_kernel(op_name, op_fn, dtype="float32"):
    """
    Dynamic multicore unary elementwise kernel.

    Args:
        x_ptr      : dtype[batch * n_cols]  input matrix, row-major
        y_ptr      : dtype[batch * n_cols]  output matrix
        batch_i32  : int32                  number of rows
        n_cols_i32 : int32                  elements per row; must be <= elements_per_tile

    Semantics:
        y[r, c] = op(x[r, c])
    """
    _meta_data = lambda: meta_data(dtype=dtype)

    @to_ir_module(meta_data=_meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        y_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(elements_per_tile)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            cid = pto.get_block_idx()
            sub_bid = pto.get_subblock_idx()
            sub_bnum = pto.get_subblock_num()
            num_blocks = pto.get_block_num()

            vid = s.index_cast(cid * sub_bnum + sub_bid)
            num_cores = s.index_cast(num_blocks * sub_bnum)

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = vid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, batch)
            num_rows = row_end - row_start

            total_elems = batch * n_cols
            tv_x = pto.as_tensor(
                tensor_type, ptr=x_ptr, shape=[total_elems], strides=[c1]
            )
            tv_y = pto.as_tensor(
                tensor_type, ptr=y_ptr, shape=[total_elems], strides=[c1]
            )

            with pto.if_context(num_rows > c0):
                tb_x = pto.alloc_tile(tile_type, valid_col=n_cols)
                tb_y = pto.alloc_tile(tile_type, valid_col=n_cols)

                for row_i in pto.for_range(c0, num_rows, c1):
                    gm_offset = (row_start + row_i) * n_cols

                    sv_x = pto.slice_view(
                        subtensor_type,
                        source=tv_x,
                        offsets=[gm_offset],
                        sizes=[n_cols],
                    )
                    sv_y = pto.slice_view(
                        subtensor_type,
                        source=tv_y,
                        offsets=[gm_offset],
                        sizes=[n_cols],
                    )

                    pto.load(sv_x, tb_x)
                    op_fn(tb_x, tb_y)
                    pto.store(tb_y, sv_y)

    _ = op_name
    return _kernel
