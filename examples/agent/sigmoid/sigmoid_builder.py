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


def build_sigmoid(fn_name="sigmoid_fp16"):
    """
    Build a dynamic-batch Sigmoid kernel in PTO DSL.

    Computes y = 1 / (1 + exp(-x)), where:
        -x        = 0 - x           (tile.sub)
        exp(-x)                      (tile.exp)
        1 + exp(-x)                  (tile.add with ones tile)
        1 / (1 + exp(-x))           (tile.reciprocal)

    Constants (1.0) are derived from the input tile itself using
    the identity exp(x - x) = exp(0) = 1.0, which avoids the need for
    scalar-tile broadcast operations not available in PTO DSL.

    UB tile budget (fp16, 4 tiles x 32 KB = 128 KB < 192 KB):
        tb_x    : input row x
        tb_ones : constant 1.0 (recomputed each row via exp(x-x))
        tb_tmp1 : intermediate / final output
        tb_tmp2 : intermediate (zeros / neg_x)

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
            with pto.if_context(n_cols > c0):
                with pto.if_context(c_tile >= n_cols):
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

                            # Derive ones: exp(x - x) = exp(0) = 1.0
                            tile.sub(tb_x, tb_x, tb_tmp2)      # tmp2 = 0.0
                            tile.exp(tb_tmp2, tb_ones)          # ones = 1.0

                            # sigmoid(x) = 1 / (1 + exp(-x))
                            tile.sub(tb_tmp2, tb_x, tb_tmp1)    # tmp1 = -x  (0 - x)
                            tile.exp(tb_tmp1, tb_tmp1)           # tmp1 = exp(-x)
                            tile.add(tb_tmp1, tb_ones, tb_tmp1)  # tmp1 = 1 + exp(-x)
                            tile.div(tb_ones, tb_tmp1, tb_tmp1)  # tmp1 = 1 / (1 + exp(-x))

                            pto.store(tb_tmp1, sv_y)

    _ = fn_name
    return _kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn-name",
        default="sigmoid_fp16",
        help="Generated kernel function name.",
    )
    args = parser.parse_args()
    print(build_sigmoid(fn_name=args.fn_name))
