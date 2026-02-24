from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const

DTYPES = {
    "float32": lambda: pto.float32,
    "float16": lambda: pto.float16,
    "int32": lambda: pto.int32,
    "int16": lambda: pto.int16,
}


def meta_data(dtype=None, tile_length=32):
    if dtype is None:
        dtype = "float32"
    if isinstance(dtype, str):
        dtype = DTYPES[dtype]()

    i32 = pto.int32
    ptr_type = pto.PtrType(dtype)
    ptr_i32 = pto.PtrType(i32)

    tensor_type = pto.TensorType(rank=1, dtype=dtype)
    tensor_i32 = pto.TensorType(rank=1, dtype=i32)

    subtensor_type = pto.SubTensorType(shape=[1, tile_length], dtype=dtype)
    subtensor_i32 = pto.SubTensorType(shape=[1, tile_length], dtype=i32)

    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[1, tile_length],
        valid_shape=[1, tile_length],
        dtype=dtype,
        memory_space="VEC",
        config=tile_cfg,
    )
    tile_i32 = pto.TileBufType(
        shape=[1, tile_length],
        valid_shape=[1, tile_length],
        dtype=i32,
        memory_space="VEC",
        config=tile_cfg,
    )

    return {
        "ptr_type": ptr_type,
        "ptr_i32": ptr_i32,
        "index_dtype": i32,
        "tensor_type": tensor_type,
        "tensor_i32": tensor_i32,
        "subtensor_type": subtensor_type,
        "subtensor_i32": subtensor_i32,
        "tile_type": tile_type,
        "tile_i32": tile_i32,
        "tile_length": tile_length,
    }


def build_gather_kernel(
    fn_name="vec_gather_2d_dynamic_float32_P1111",
    dtype="float32",
    tile_length=32,
    mask_pattern="P1111",
):
    """
    1D dynamic multicore gather, same tiling logic as vec_add_1d_dynamic.

    Per-tile semantics:
      for each tile starting at base:
        out[base + j] = src[base + indices[base + j]]

    IMPORTANT:
      - indices are *within-tile* indices, must be in [0, tile_length-1]
      - argN is total elements
      - works best when N is multiple of tile_length (no tail handling other than dropping extra tiles)
    """
    assert mask_pattern in ("P1111", "P0101", "P0001"), f"Unsupported mask_pattern: {mask_pattern}"
    _meta_data = lambda: meta_data(dtype=dtype, tile_length=tile_length)

    def _kernel(
        arg0: "ptr_type",    # src (T*)
        arg1: "ptr_i32",     # indices (i32*) values in [0..tile_length-1] per tile
        arg2: "ptr_type",    # out (T*)
        argN: "index_dtype", # total elements
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(tile_length)

        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        vid = cid * sub_bnum + sub_bid
        num_blocks = pto.get_block_num()

        vid_idx = pto.index_cast(vid)
        num_cores = pto.index_cast(num_blocks)
        total_elements = pto.index_cast(argN)

        num_tiles_global = pto.ceil_div(total_elements, c_tile)
        num_tiles_per_core = pto.ceil_div(num_tiles_global, num_cores)
        tile_offset_this_core = vid_idx * num_tiles_per_core

        with pto.vector_section():
            tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[total_elements], strides=[c1])
            tv1 = pto.as_tensor(tensor_i32, ptr=arg1, shape=[total_elements], strides=[c1])
            tv2 = pto.as_tensor(tensor_type, ptr=arg2, shape=[total_elements], strides=[c1])

            tb_src = pto.alloc_tile(tile_type)
            tb_idx = pto.alloc_tile(tile_i32)
            tb_tmp = pto.alloc_tile(tile_type)
            tb_out = pto.alloc_tile(tile_type)

            # Skip whole core if its starting tile is already out-of-bound.
            with pto.if_context(tile_offset_this_core < num_tiles_global):
                tiles_end_this_core = tile_offset_this_core + num_tiles_per_core
                need_truncate = tiles_end_this_core > num_tiles_global
                remaining_tiles = num_tiles_global - tile_offset_this_core

                def _process_tiles(tiles_to_process):
                    elements_to_process = tiles_to_process * c_tile
                    with pto.if_context(elements_to_process > c0):
                        for i in pto.for_range(c0, tiles_to_process, c1):
                            tile_offset_global = i + tile_offset_this_core
                            offset_global = tile_offset_global * c_tile

                            sv0 = pto.slice_view(
                                subtensor_type, source=tv0,
                                offsets=[offset_global], sizes=[c_tile]
                            )
                            sv1 = pto.slice_view(
                                subtensor_i32, source=tv1,
                                offsets=[offset_global], sizes=[c_tile]
                            )
                            sv2 = pto.slice_view(
                                subtensor_type, source=tv2,
                                offsets=[offset_global], sizes=[c_tile]
                            )

                            pto.load(sv0, tb_src)
                            pto.load(sv1, tb_idx)

                            # gather within tile by indices
                            pto.gather(tb_src, tb_tmp, tb_idx)

                            # optional mask gather
                            pto.gather(tb_tmp, tb_out, mask_pattern=mask_pattern)

                            pto.store(tb_out, sv2)

                def _truncated():
                    _process_tiles(remaining_tiles)

                def _full():
                    _process_tiles(num_tiles_per_core)

                # ✅ statement-style conditional (no SSA result), like your matmul example
                pto.cond(need_truncate, _truncated, _full)

    _kernel.__name__ = fn_name
    return to_ir_module(meta_data=_meta_data)(_kernel)


if __name__ == "__main__":
    # example
    print(build_gather_kernel(dtype="float32", tile_length=32, mask_pattern="P1111"))