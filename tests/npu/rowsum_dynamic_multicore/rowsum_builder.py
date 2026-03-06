from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const

# 32 KB of UB
_TILE_SIZE_BYTES = 32 * 1024
_DTYPE_BYTES = {"fp16": 2, "fp32": 4}


def meta_data(dtype="fp32"):
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


def build_rowsum(fn_name="rowsum_fp32", dtype="fp32"):
    """
    Computes per-row sum across columns using PTO TROWSUM (pto.row_sum wrapper).

    Args:
        x_ptr : dtype[batch * n_cols]    input matrix flattened row-major
        y_ptr : dtype[batch]             output vector, one sum per row
        batch : int32
        n_cols: int32 (<= elements_per_tile)

    Semantics:
        y[row] = sum_{j=0..n_cols-1} x[row, j]
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

        batch = pto.index_cast(batch_i32)
        n_cols = pto.index_cast(n_cols_i32)

        with pto.vector_section():
            bid = pto.index_cast(pto.get_block_idx())
            num_cores = pto.index_cast(pto.get_block_num())

            rows_per_core = pto.ceil_div(batch, num_cores)
            row_start = bid * rows_per_core
            row_end = pto.min_u(row_start + rows_per_core, batch)
            num_rows = row_end - row_start

            total_elems = batch * n_cols
            tv_x = pto.as_tensor(
                tensor_type, ptr=x_ptr, shape=[total_elems], strides=[c1]
            )
            tv_y = pto.as_tensor(
                tensor_type, ptr=y_ptr, shape=[batch], strides=[c1]
            )

            with pto.if_context(num_rows > c0):
                tb_x = pto.alloc_tile(tile_type, valid_col=n_cols)
                tb_sum = pto.alloc_tile(
                    tile_type, valid_col=c1
                )  # scalar output
                tb_tmp = pto.alloc_tile(
                    tile_type, valid_col=n_cols
                )  # scratch

                for r in pto.for_range(c0, num_rows, c1):
                    gm_offset = (row_start + r) * n_cols

                    sv_x = pto.slice_view(
                        subtensor_in,
                        source=tv_x,
                        offsets=[gm_offset],
                        sizes=[n_cols],
                    )

                    # y is a vector of length batch; write one element per row
                    sv_y = pto.slice_view(
                        subtensor_in,
                        source=tv_y,
                        offsets=[row_start + r],
                        sizes=[c1],
                    )

                    pto.load(sv_x, tb_x)
                    pto.row_sum(tb_x, tb_tmp, tb_sum)

                    # Store the 1-element tile to y[row]
                    pto.store(tb_sum, sv_y)

    _ = fn_name
    return _kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fn-name", default="rowsum_fp32")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()
    print(build_rowsum(fn_name=args.fn_name, dtype=args.dtype))
