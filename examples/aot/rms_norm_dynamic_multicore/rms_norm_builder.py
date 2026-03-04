from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const

# 32 KB of UB / sizeof(fp16) = 16384 elements per tile
ELEMENTS_PER_TILE = 32 * 1024 // 2


def meta_data():
    dtype = pto.float16
    ptr_type = pto.PtrType(dtype)
    index_dtype = pto.int32

    tensor_type = pto.TensorType(rank=1, dtype=dtype)
    subtensor_type = pto.SubTensorType(shape=[1, ELEMENTS_PER_TILE], dtype=dtype)

    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[1, ELEMENTS_PER_TILE],
        valid_shape=[1, -1],
        dtype=dtype,
        memory_space="VEC",
        config=tile_cfg,
    )

    return {
        "ptr_type": ptr_type,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_type": subtensor_type,
        "tile_type": tile_type,
    }


def build_rms_norm(fn_name="rms_norm_fp16"):
    """
    Build a dynamic-batch RMS Norm kernel in PTO DSL.

    Computes y = x * rsqrt(mean(x²)) * w, where:
        mean(x²) = sum(x_i²) / n_cols  (no eps; fp16 1e-6 underflows on most NPU hardware)

    UB tile budget (fp16, 5 tiles × 32 KB = 160 KB < 192 KB UB):
        tb_x   : input row
        tb_w   : weight (loaded once, reused across rows)
        tb_x2  : x² intermediate, then x_normed, then output y
        tb_sum : row-sum result (valid_col=1 → per-row scalar rstd)
        tb_tmp : row-sum working buffer

    Kernel args:
        x_ptr   : fp16[batch * n_cols]   -- input
        w_ptr   : fp16[n_cols]           -- per-element weight
        y_ptr   : fp16[batch * n_cols]   -- output
        batch   : int32                  -- number of rows
        n_cols  : int32                  -- elements per row; must be <= 16384
    """

    @to_ir_module(meta_data=meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        w_ptr: "ptr_type",
        y_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(ELEMENTS_PER_TILE)

        batch = pto.index_cast(batch_i32)
        n_cols = pto.index_cast(n_cols_i32)

        with pto.vector_section():
            # Guard: n_cols must be in (0, ELEMENTS_PER_TILE].
            with pto.if_context(n_cols > c0):
                with pto.if_context(c_tile >= n_cols):
                    bid = pto.index_cast(pto.get_block_idx())
                    num_cores = pto.index_cast(pto.get_block_num())

                    # Distribute rows across cores.
                    rows_per_core = pto.ceil_div(batch, num_cores)
                    row_start = bid * rows_per_core
                    row_end = pto.min_u(row_start + rows_per_core, batch)
                    num_rows = row_end - row_start

                    total_elems = batch * n_cols
                    tv_x = pto.as_tensor(
                        tensor_type, ptr=x_ptr, shape=[total_elems], strides=[c1]
                    )
                    tv_w = pto.as_tensor(
                        tensor_type, ptr=w_ptr, shape=[n_cols], strides=[c1]
                    )
                    tv_y = pto.as_tensor(
                        tensor_type, ptr=y_ptr, shape=[total_elems], strides=[c1]
                    )

                    with pto.if_context(num_rows > c0):
                        # Allocate 5 UB tiles (160 KB total, well under 192 KB UB).
                        tb_x = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_w = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_x2 = pto.alloc_tile(tile_type, valid_col=n_cols)
                        # TRowSumOp broadcasts the per-row sum to all valid_col positions.
                        tb_sum = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_tmp = pto.alloc_tile(tile_type, valid_col=n_cols)

                        # Precompute scalar 1/n_cols for mean normalisation.
                        n_cols_f16 = pto.index_to_float(n_cols, pto.float16)

                        # Load weight once (shared across all rows).
                        sv_w = pto.slice_view(
                            subtensor_type,
                            source=tv_w,
                            offsets=[c0],
                            sizes=[n_cols],
                        )
                        pto.load(sv_w, tb_w)

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

                            # x² = x * x
                            pto.mul(tb_x, tb_x, tb_x2)

                            # sum(x²) → tb_sum (valid_col=1, one scalar per row)
                            pto.row_sum(tb_x2, tb_tmp, tb_sum)

                            # mean(x²) = sum(x²) / n_cols
                            pto.div_s(tb_sum, n_cols_f16, tb_sum)

                            # rstd = rsqrt(mean(x²))  (stored in tb_sum, valid_col=1)
                            pto.rsqrt(tb_sum, tb_sum)

                            # x_normed[i] = x[i] * rstd  (tb_sum has rstd in all cols)
                            pto.mul(tb_x, tb_sum, tb_x2)

                            # y = x_normed * w
                            pto.mul(tb_x2, tb_w, tb_x2)

                            pto.store(tb_x2, sv_y)

    _ = fn_name
    return _kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn-name",
        default="rms_norm_fp16",
        help="Generated kernel function name.",
    )
    args = parser.parse_args()
    print(build_rms_norm(fn_name=args.fn_name))
