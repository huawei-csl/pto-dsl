from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

# TSORT32 sorts within fixed 32-element blocks.
# Each input element expands into (score, index) pairs in the output:
#   float16: 4 float16 words  [score_f16, zero, idx_lo_u16, idx_hi_u16]
#   float32: 2 float32 words  [score_f32, idx_u32]
_SORT_BLOCK_LEN = 32

_DTYPES = {
    "float16": lambda: pto.float16,
    "float32": lambda: pto.float32,
}

# Output words per input element (in units of the src dtype)
_DST_STRIDE = {
    "float16": 4,
    "float32": 2,
}


def meta_data(dtype="float16", tile_length=1024):
    if isinstance(dtype, str):
        dtype_str = dtype
        pto_dtype = _DTYPES[dtype]()
    else:
        pto_dtype = dtype
        dtype_str = "float16"

    dst_stride = _DST_STRIDE[dtype_str]
    u32 = pto.uint32
    dst_tile_length = tile_length * dst_stride

    tile_cfg = pto.TileBufConfig()
    return {
        "ptr_src": pto.PtrType(pto_dtype),
        "ptr_u32": pto.PtrType(u32),
        "ptr_dst": pto.PtrType(pto_dtype),
        "index_dtype": pto.int32,
        "tensor_src": pto.TensorType(rank=2, dtype=pto_dtype),
        "tensor_u32": pto.TensorType(rank=2, dtype=u32),
        "tensor_dst": pto.TensorType(rank=2, dtype=pto_dtype),
        "subtensor_src": pto.SubTensorType(shape=[1, tile_length], dtype=pto_dtype),
        "subtensor_u32": pto.SubTensorType(shape=[1, tile_length], dtype=u32),
        "subtensor_dst": pto.SubTensorType(shape=[1, dst_tile_length], dtype=pto_dtype),
        "tile_src": pto.TileBufType(
            shape=[1, tile_length],
            valid_shape=[1, tile_length],
            dtype=pto_dtype,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_u32": pto.TileBufType(
            shape=[1, tile_length],
            valid_shape=[1, tile_length],
            dtype=u32,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_dst": pto.TileBufType(
            shape=[1, dst_tile_length],
            valid_shape=[1, dst_tile_length],
            dtype=pto_dtype,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_length": tile_length,
        "dst_tile_length": dst_tile_length,
    }


def build_tsort32_kernel(
    fn_name="tsort32_1d_dynamic_float16",
    dtype="float16",
    tile_length=1024,
):
    """Build a 1D dynamic multicore TSORT32 kernel.

    For each tile of tile_length elements:
      - Reads src (scores) and idx (uint32 indices).
      - Calls TSORT32, which sorts within _SORT_BLOCK_LEN-element blocks and
        writes interleaved (score, index) pairs to dst.
      - dst is dst_stride times wider than src in same-dtype words:
          float16: dst_stride=4  →  dst is float16[N * 4]
          float32: dst_stride=2  →  dst is float32[N * 2]

    Constraints:
      - tile_length must be a multiple of _SORT_BLOCK_LEN (32)
      - N (total input elements) must be a multiple of tile_length
    """
    assert (
        tile_length % _SORT_BLOCK_LEN == 0
    ), f"tile_length must be a multiple of {_SORT_BLOCK_LEN}, got {tile_length}"
    dtype_str = dtype if isinstance(dtype, str) else "float16"
    dst_stride = _DST_STRIDE[dtype_str]
    dst_tile_length = tile_length * dst_stride
    _meta_data = lambda: meta_data(dtype=dtype, tile_length=tile_length)

    def _kernel(
        arg_src: "ptr_src",  # input scores [N]
        arg_idx: "ptr_u32",  # uint32 input indices [N]
        arg_dst: "ptr_dst",  # output pairs [N * dst_stride]
        argN: "index_dtype",  # total input elements (multiple of tile_length)
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(tile_length)
        c_dst_tile = const(dst_tile_length)

        total_elements = s.index_cast(argN)
        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        vid = cid * sub_bnum + sub_bid
        num_blocks = pto.get_block_num()

        vid_idx = s.index_cast(vid)
        num_cores = s.index_cast(num_blocks * sub_bnum)

        num_tiles_global = s.ceil_div(total_elements, c_tile)
        num_tiles_per_core = s.ceil_div(num_tiles_global, num_cores)
        tile_offset_this_core = vid_idx * num_tiles_per_core

        with pto.vector_section():
            tv_src = pto.as_tensor(
                tensor_src,
                ptr=arg_src,
                shape=[num_tiles_global, c_tile],
                strides=[c_tile, c1],
            )
            tv_idx = pto.as_tensor(
                tensor_u32,
                ptr=arg_idx,
                shape=[num_tiles_global, c_tile],
                strides=[c_tile, c1],
            )
            tv_dst = pto.as_tensor(
                tensor_dst,
                ptr=arg_dst,
                shape=[num_tiles_global, c_dst_tile],
                strides=[c_dst_tile, c1],
            )

            tb_src = pto.alloc_tile(tile_src)
            tb_idx = pto.alloc_tile(tile_u32)
            tb_dst = pto.alloc_tile(tile_dst)

            with pto.if_context(tile_offset_this_core < num_tiles_global):
                tiles_end_this_core = tile_offset_this_core + num_tiles_per_core
                need_truncate = tiles_end_this_core > num_tiles_global
                remaining_tiles = num_tiles_global - tile_offset_this_core
                tiles_to_process = s.select(
                    need_truncate, remaining_tiles, num_tiles_per_core
                )

                with pto.if_context(tiles_to_process > c0):
                    for i in pto.range(c0, tiles_to_process, c1):
                        ti = i + tile_offset_this_core

                        sv_src = pto.slice_view(
                            subtensor_src,
                            source=tv_src,
                            offsets=[ti, c0],
                            sizes=[c1, c_tile],
                        )
                        sv_idx = pto.slice_view(
                            subtensor_u32,
                            source=tv_idx,
                            offsets=[ti, c0],
                            sizes=[c1, c_tile],
                        )
                        sv_dst = pto.slice_view(
                            subtensor_dst,
                            source=tv_dst,
                            offsets=[ti, c0],
                            sizes=[c1, c_dst_tile],
                        )

                        pto.load(sv_src, tb_src)
                        pto.load(sv_idx, tb_idx)
                        tile.sort32(tb_src, tb_dst, tb_idx)
                        pto.store(tb_dst, sv_dst)

    _kernel.__name__ = fn_name
    return to_ir_module(meta_data=_meta_data)(_kernel)


if __name__ == "__main__":
    import sys

    dtype = sys.argv[1] if len(sys.argv) > 1 else "float16"
    print(build_tsort32_kernel(dtype=dtype))
