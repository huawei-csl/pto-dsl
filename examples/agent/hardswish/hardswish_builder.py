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


def build_hardswish(fn_name="hardswish_fp16"):
    """
    Build a dynamic-batch Hard-Swish kernel in PTO DSL.

    Computes y = x * clamp(x + 3, 0, 6) / 6, where:
        clamp(v, 0, 6) = relu(v) - relu(v - 6)

    Constants (1.0, 3.0, 6.0) are derived from the input tile itself using
    the identity exp(a - a) = exp(0) = 1.0, which avoids the need for
    scalar-tile broadcast operations not available in PTO DSL.

    UB tile budget (fp16, 4 tiles x 32 KB = 128 KB < 192 KB):
        tb_x   : input row x
        tb_t1  : intermediate
        tb_t2  : intermediate
        tb_t3  : holds 6.0, then intermediate

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
                        tb_t1 = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_t2 = pto.alloc_tile(tile_type, valid_col=n_cols)
                        tb_t3 = pto.alloc_tile(tile_type, valid_col=n_cols)

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

                            # Load input
                            pto.load(sv_x, tb_x)

                            # Derive constants from data:
                            #   x - x = 0  =>  exp(0) = 1.0
                            tile.sub(tb_x, tb_x, tb_t1)    # tb_t1 = 0.0
                            tile.exp(tb_t1, tb_t1)          # tb_t1 = 1.0 (ones)

                            # Build 3.0:
                            tile.add(tb_t1, tb_t1, tb_t2)   # tb_t2 = 2.0
                            tile.add(tb_t2, tb_t1, tb_t2)   # tb_t2 = 3.0

                            # Build 6.0:
                            tile.add(tb_t2, tb_t2, tb_t3)   # tb_t3 = 6.0

                            # Compute x + 3:
                            tile.add(tb_x, tb_t2, tb_t2)    # tb_t2 = x + 3

                            # clamp(x+3, 0, 6) = relu(x+3) - relu(relu(x+3) - 6)
                            tile.relu(tb_t2, tb_t2)          # tb_t2 = relu(x + 3) = max(x+3, 0)
                            tile.sub(tb_t2, tb_t3, tb_t1)    # tb_t1 = max(x+3, 0) - 6
                            tile.relu(tb_t1, tb_t1)           # tb_t1 = relu(max(x+3,0) - 6)
                            tile.sub(tb_t2, tb_t1, tb_t1)    # tb_t1 = clamp(x+3, 0, 6)

                            # y = x * clamp(x+3, 0, 6) / 6
                            tile.mul(tb_x, tb_t1, tb_t1)     # tb_t1 = x * clamp(x+3, 0, 6)
                            tile.div(tb_t1, tb_t3, tb_t1)    # tb_t1 = y

                            pto.store(tb_t1, sv_y)

    _ = fn_name
    return _kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fn-name",
        default="hardswish_fp16",
        help="Generated kernel function name.",
    )
    args = parser.parse_args()
    print(build_hardswish(fn_name=args.fn_name))

