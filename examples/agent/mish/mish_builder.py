from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

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


def build_mish(fn_name="mish_fp16"):
    """
    Build a dynamic-batch Mish kernel in PTO DSL.

    Computes y = x * tanh(softplus(x)), where:
        softplus(x) = ln(1 + exp(x))
        tanh(softplus(x)) = ((1+exp(x))^2 - 1) / ((1+exp(x))^2 + 1)

    This avoids ln() by using the algebraic identity:
        exp(2*ln(u)) = u^2

    Constants (1.0) are derived from the input tile itself using
    the identity exp(x - x) = exp(0) = 1.0, which avoids the need for
    scalar-tile broadcast operations not available in PTO DSL.

    UB tile budget (fp16, 4 tiles x 32 KB = 128 KB < 192 KB):
        tb_x    : input row x
        tb_ones : constant 1.0 (recomputed each row via exp(x-x))
        tb_tmp1 : intermediate / final output
        tb_tmp2 : intermediate

    Kernel args:
        x_ptr   : fp16[batch * n_cols]  -- input
        y_ptr   : fp16[batch * n_cols]  -- output
        batch   : int32                 -- number of rows
        n_cols  : int32                 -- elements per row; must be <= 16384
    """

    @to_ir_module(meta_data=meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        y_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(ELEMENTS_PER_TILE)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            # Guard: n_cols must be in (0, ELEMENTS_PER_TILE].
            with pto.if_context(n_cols > c0):
                with pto.if_context(c_tile >= n_cols):
                    cid = pto.get_block_idx()
                    sub_bid = pto.get_subblock_idx()
                    sub_bnum = pto.get_subblock_num()
                    num_blocks = pto.get_block_num()

                    vid = s.index_cast(cid * sub_bnum + sub_bid)
                    num_cores = s.index_cast(num_blocks * sub_bnum)

                    # Distribute rows across cores (row-level parallelism).
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
                        # Allocate 4 UB tiles (128 KB total, well under 192 KB UB).
                        tb_x = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_ones = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_tmp1 = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_tmp2 = pto.alloc_tile(tile_type, valid_col=n_cols)

                        for row_i in pto.range(c0, num_rows, c1):
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

                            # Derive constant 1.0 from data:
                            #   x - x = 0  =>  exp(0) = 1.0
                            tile.sub(tb_x, tb_x, tb_tmp1)    # tmp1 = 0.0
                            tile.exp(tb_tmp1, tb_ones)        # ones = 1.0

                            # Compute mish(x) = x * tanh(softplus(x))
                            # Using identity: tanh(ln(1+exp(x))) = ((1+exp(x))^2 - 1) / ((1+exp(x))^2 + 1)

                            # Step 1: exp(x)
                            tile.exp(tb_x, tb_tmp1)           # tmp1 = exp(x)

                            # Step 2: u = 1 + exp(x)
                            tile.add(tb_ones, tb_tmp1, tb_tmp1)  # tmp1 = 1 + exp(x)

                            # Step 3: u^2 = (1 + exp(x))^2
                            tile.mul(tb_tmp1, tb_tmp1, tb_tmp2)  # tmp2 = (1+exp(x))^2

                            # Step 4: numerator = u^2 - 1
                            tile.sub(tb_tmp2, tb_ones, tb_tmp1)  # tmp1 = (1+exp(x))^2 - 1

                            # Step 5: denominator = u^2 + 1
                            tile.add(tb_tmp2, tb_ones, tb_tmp2)  # tmp2 = (1+exp(x))^2 + 1

                            # Step 6: tanh(softplus(x)) = num / den
                            tile.div(tb_tmp1, tb_tmp2, tb_tmp1)  # tmp1 = tanh(softplus(x))

                            # Step 7: mish(x) = x * tanh(softplus(x))
                            tile.mul(tb_x, tb_tmp1, tb_tmp1)     # tmp1 = mish(x)

                            pto.store(tb_tmp1, sv_y)

    _ = fn_name
    return _kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn-name",
        default="mish_fp16",
        help="Generated kernel function name.",
    )
    args = parser.parse_args()
    print(build_mish(fn_name=args.fn_name))
