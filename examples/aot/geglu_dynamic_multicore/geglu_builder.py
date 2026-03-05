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


def build_geglu(fn_name="geglu_fp16"):
    """
    Build a dynamic-batch GEGLU kernel in PTO DSL.

    Computes c = gelu_approx(a) * b, where:
        gelu_approx(a) = 0.5 * a * (1 + tanh(a))
        tanh(a)        = (exp(2a) - 1) / (exp(2a) + 1)

    Constants (1.0, 2.0) are derived from the input tile itself using
    the identity exp(a - a) = exp(0) = 1.0, which avoids the need for
    scalar-tile broadcast operations not available in PTO DSL.

    UB tile budget (fp16, 5 tiles × 32 KB = 160 KB < 192 KB):
        tb_a    : input row a
        tb_b    : input row b
        tb_ones : constant 1.0 (recomputed each row via exp(a-a))
        tb_tmp1 : intermediate / final output
        tb_tmp2 : intermediate

    Kernel args:
        a_ptr   : fp16[batch * n_cols]  -- gating input
        b_ptr   : fp16[batch * n_cols]  -- linear input
        c_ptr   : fp16[batch * n_cols]  -- output
        batch   : int32                 -- number of rows
        n_cols  : int32                 -- elements per row; must be <= 16384
    """

    @to_ir_module(meta_data=meta_data)
    def _kernel(
        a_ptr: "ptr_type",
        b_ptr: "ptr_type",
        c_ptr: "ptr_type",
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
                    cid = pto.get_block_idx()
                    sub_bid = pto.get_subblock_idx()
                    sub_bnum = pto.get_subblock_num()
                    num_blocks = pto.get_block_num()

                    vid = pto.index_cast(cid * sub_bnum + sub_bid)  # vector core index
                    num_cores = pto.index_cast(num_blocks * sub_bnum)  # number of vector cores

                    # Distribute rows across cores (row-level parallelism).
                    rows_per_core = pto.ceil_div(batch, num_cores)
                    row_start = vid * rows_per_core
                    row_end = pto.min_u(row_start + rows_per_core, batch)
                    num_rows = row_end - row_start

                    total_elems = batch * n_cols
                    tv_a = pto.as_tensor(
                        tensor_type, ptr=a_ptr, shape=[total_elems], strides=[c1]
                    )
                    tv_b = pto.as_tensor(
                        tensor_type, ptr=b_ptr, shape=[total_elems], strides=[c1]
                    )
                    tv_c = pto.as_tensor(
                        tensor_type, ptr=c_ptr, shape=[total_elems], strides=[c1]
                    )

                    with pto.if_context(num_rows > c0):
                        # Allocate 5 UB tiles (160 KB total, well under 192 KB UB).
                        tb_a = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_b = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_ones = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_tmp1 = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_tmp2 = pto.alloc_tile(tile_type, valid_col=n_cols)

                        for row_i in pto.for_range(c0, num_rows, c1):
                            gm_offset = (row_start + row_i) * n_cols

                            sv_a = pto.slice_view(
                                subtensor_type,
                                source=tv_a,
                                offsets=[gm_offset],
                                sizes=[n_cols],
                            )
                            sv_b = pto.slice_view(
                                subtensor_type,
                                source=tv_b,
                                offsets=[gm_offset],
                                sizes=[n_cols],
                            )
                            sv_c = pto.slice_view(
                                subtensor_type,
                                source=tv_c,
                                offsets=[gm_offset],
                                sizes=[n_cols],
                            )

                            pto.load(sv_a, tb_a)
                            pto.load(sv_b, tb_b)

                            # Derive constants from data (no scalar-tile broadcast needed):
                            #   a - a = 0  =>  exp(0) = 1.0
                            pto.sub(tb_a, tb_a, tb_tmp2)  # tmp2 = 0.0
                            pto.exp(tb_tmp2, tb_ones)  # ones = 1.0

                            # tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
                            pto.add(tb_a, tb_a, tb_tmp1)  # tmp1 = 2a
                            pto.exp(tb_tmp1, tb_tmp1)  # tmp1 = exp(2a)
                            pto.sub(tb_tmp1, tb_ones, tb_tmp2)  # tmp2 = exp(2a) - 1
                            pto.add(tb_tmp1, tb_ones, tb_tmp1)  # tmp1 = exp(2a) + 1
                            pto.div(tb_tmp2, tb_tmp1, tb_tmp2)  # tmp2 = tanh(a)

                            # gelu_approx(a) = a * (1 + tanh(a)) / 2
                            pto.add(tb_ones, tb_tmp2, tb_tmp1)  # tmp1 = 1 + tanh(a)
                            pto.mul(tb_a, tb_tmp1, tb_tmp1)  # tmp1 = a * (1 + tanh(a))
                            pto.add(tb_ones, tb_ones, tb_tmp2)  # tmp2 = 2.0
                            pto.div(tb_tmp1, tb_tmp2, tb_tmp1)  # tmp1 = gelu_approx(a)

                            # GEGLU: c = gelu_approx(a) * b
                            pto.mul(tb_tmp1, tb_b, tb_tmp1)  # tmp1 = c
                            pto.store(tb_tmp1, sv_c)

    _ = fn_name
    return _kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn-name",
        default="geglu_fp16",
        help="Generated kernel function name.",
    )
    args = parser.parse_args()
    print(build_geglu(fn_name=args.fn_name))
