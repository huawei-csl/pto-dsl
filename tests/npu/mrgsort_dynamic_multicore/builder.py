from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

DTYPES = {
    "float32": lambda: pto.float32,
    "float16": lambda: pto.float16,
}

# TMRGSORT's blockLen parameter is in float32-word units:
#   hw_block_len = block_len * (sizeof(float) / sizeof(T))
_TYPE_COEF = {"float32": 1, "float16": 2}


def meta_data(dtype=None, tile_length=1024):
    if dtype is None:
        dtype = "float32"
    if isinstance(dtype, str):
        dtype = DTYPES[dtype]()

    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    # 2D tensor view: shape [num_tiles, tile_length], matching the expand_builder pattern.
    tensor_type = pto.TensorType(rank=2, dtype=dtype)
    subtensor_type = pto.SubTensorType(shape=[1, tile_length], dtype=dtype)
    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[1, tile_length],
        valid_shape=[1, tile_length],
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
        "tile_length": tile_length,
    }


def build_mrgsort_kernel(
    fn_name="vec_mrgsort_1d_dynamic_float32",
    dtype="float32",
    tile_length=1024,
    block_len=32,
):
    """Build a 1D dynamic multicore merge-sort kernel.

    Each tile of tile_length elements is treated as containing
    tile_length // block_len pre-sorted sub-lists of block_len elements.
    TMRGSORT merges groups of 4 sub-lists (block_len*4 elements) independently;
    repeatTimes = tile_length // (block_len * 4) such groups per tile.

    The hardware blockLen passed to TMRGSORT is scaled by TYPE_COEF
    (= sizeof(float) / sizeof(T)) per the instruction's float32-word semantics:
      hw_block_len = block_len * TYPE_COEF

    Constraints (enforced by TMRGSORT):
      - hw_block_len must be a multiple of 64
      - tile_length must be a multiple of hw_block_len * 4
      - repeatTimes = tile_length / (hw_block_len * 4) must be in [1, 255]
    """
    dtype_str = dtype if isinstance(dtype, str) else "float32"
    hw_block_len = block_len * _TYPE_COEF.get(dtype_str, 1)
    _meta_data = lambda: meta_data(dtype=dtype, tile_length=tile_length)

    def _kernel(
        arg0: "ptr_type",  # src: input with sorted sub-lists
        arg1: "ptr_type",  # out: merged sorted output
        argN: "index_dtype",  # total number of elements (multiple of tile_length)
    ) -> None:
        assert tile_length % (hw_block_len * 4) == 0
        assert hw_block_len % 64 == 0
        c0 = const(0)
        c1 = const(1)
        c_tile = const(tile_length)

        total_elements = s.index_cast(argN)
        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        vid = cid * sub_bnum + sub_bid
        num_blocks = pto.get_block_num()

        vid_idx = s.index_cast(vid)
        # Total virtual cores = num_blocks * subblock_num (matches add_dynamic_multicore).
        num_cores = s.index_cast(num_blocks * sub_bnum)

        num_tiles_global = s.ceil_div(total_elements, c_tile)
        num_tiles_per_core = s.ceil_div(num_tiles_global, num_cores)
        tile_offset_this_core = vid_idx * num_tiles_per_core

        with pto.vector_section():
            # 2D tensor views: shape=[num_tiles, tile_length], strides=[tile_length, 1].
            # Mirrors the expand_builder layout where rows are tiles and columns are elements.
            tv0 = pto.as_tensor(
                tensor_type,
                ptr=arg0,
                shape=[num_tiles_global, c_tile],
                strides=[c_tile, c1],
            )
            tv1 = pto.as_tensor(
                tensor_type,
                ptr=arg1,
                shape=[num_tiles_global, c_tile],
                strides=[c_tile, c1],
            )

            tb_src = pto.alloc_tile(tile_type)
            tb_tmp = pto.alloc_tile(tile_type)
            tb_dst = pto.alloc_tile(tile_type)

            with pto.if_context(tile_offset_this_core < num_tiles_global):
                tiles_end_this_core = tile_offset_this_core + num_tiles_per_core
                need_truncate = tiles_end_this_core > num_tiles_global
                remaining_tiles = num_tiles_global - tile_offset_this_core
                tiles_to_process = s.select(
                    need_truncate, remaining_tiles, num_tiles_per_core
                )

                with pto.if_context(tiles_to_process > c0):
                    for i in pto.range(c0, tiles_to_process, c1):
                        tile_idx = i + tile_offset_this_core

                        sv0 = pto.slice_view(
                            subtensor_type,
                            source=tv0,
                            offsets=[tile_idx, c0],
                            sizes=[c1, c_tile],
                        )

                        pto.load(sv0, tb_src)
                        # Multi-pass merge sort: blockLen doubles each pass (reference
                        # MrgsortSingleRow pattern). This loop is unrolled at code-gen
                        # time since all bounds are Python-level constants.
                        cur_block_len = hw_block_len
                        while cur_block_len * 4 <= tile_length:
                            tile.mrgsort(tb_src, tb_tmp, const(cur_block_len))
                            tile.mov(tb_tmp, tb_src)
                            cur_block_len *= 4
                        tile.mov(tb_src, tb_dst)

                        sv1 = pto.slice_view(
                            subtensor_type,
                            source=tv1,
                            offsets=[tile_idx, c0],
                            sizes=[c1, c_tile],
                        )
                        pto.store(tb_dst, sv1)

    _kernel.__name__ = fn_name
    return to_ir_module(meta_data=_meta_data)(_kernel)


if __name__ == "__main__":
    print(build_mrgsort_kernel(dtype="float32", tile_length=1024, block_len=64))
