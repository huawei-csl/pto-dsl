from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const

DTYPES = {
    "float32": lambda: pto.float32,
    "float16": lambda: pto.float16,
    "int32": lambda: pto.int32,
    "int16": lambda: pto.int16,
}

TILE_LENGTH = 32
TILE_HALF   = TILE_LENGTH // 2


def meta_data(dtype=None):
    if dtype is None:
        dtype = "float32"
    if isinstance(dtype, str):
        dtype = DTYPES[dtype]()

    tile_cfg = pto.TileBufConfig()
    i32 = pto.int32

    ptr_type = pto.PtrType(dtype)
    tensor_type    = pto.TensorType(rank=1, dtype=dtype)
    subtensor_tile = pto.SubTensorType(shape=[1, TILE_LENGTH], dtype=dtype)
    subtensor_half = pto.SubTensorType(shape=[1, TILE_HALF],   dtype=dtype)
    tile_row = pto.TileBufType(
        shape=[1, TILE_LENGTH], valid_shape=[1, TILE_LENGTH],
        dtype=dtype, memory_space="VEC", config=tile_cfg,
    )
    tile_half = pto.TileBufType(
        shape=[1, TILE_HALF], valid_shape=[1, TILE_HALF],
        dtype=dtype, memory_space="VEC", config=tile_cfg,
    )

    return {
        "ptr_type":      ptr_type,
        "index_dtype":   i32,
        "tensor_type":   tensor_type,
        "subtensor_tile": subtensor_tile,
        "subtensor_half": subtensor_half,
        "tile_row":  tile_row,
        "tile_half": tile_half,
    }


def build_hadamard_kernel_dynamic(fn_name="hadamard_kernel_dynamic", dtype=None):
    """Hadamard kernel tiled at TILE_LENGTH=32 using P0101/P1010 gather patterns.

    Kernel arguments:
      arg0    – src ptr
      arg1    – out ptr
      argRows – number of rows  (int32, runtime)
      argCols – number of cols  (int32, runtime, must be a power-of-2 multiple of TILE_LENGTH)
      argLog  – log2(argCols)  (int32, runtime)
    """
    _meta_data = lambda: meta_data(dtype)

    def _kernel(
        arg0:    "ptr_type",
        arg1:    "ptr_type",
        argRows: "index_dtype",
        argCols: "index_dtype",
        argLog:  "index_dtype",
    ) -> None:
        c0    = const(0)
        c1    = const(1)
        ctile = const(TILE_LENGTH)
        chalf = const(TILE_HALF)

        crow = pto.index_cast(argRows)
        ccol = pto.index_cast(argCols)
        clog = pto.index_cast(argLog)  # log2(cols); cols must be a power-of-2 multiple of TILE_LENGTH

        cid      = pto.get_block_idx()
        sub_bid  = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        vid      = cid * sub_bnum + sub_bid
        num_blocks = pto.get_block_num()

        vid_idx   = pto.index_cast(vid)
        num_cores = pto.index_cast(num_blocks)

        total_elements  = crow * ccol
        tiles_per_row   = pto.ceil_div(ccol, ctile)
        rows_per_core   = pto.ceil_div(crow, num_cores)
        row_offset_this_core = vid_idx * rows_per_core

        with pto.vector_section():
            tv_src = pto.as_tensor(tensor_type, ptr=arg0, shape=[total_elements], strides=[c1])
            tv_out = pto.as_tensor(tensor_type, ptr=arg1, shape=[total_elements], strides=[c1])

            evenTile = pto.alloc_tile(tile_half)
            oddTile  = pto.alloc_tile(tile_half)
            sumTile  = pto.alloc_tile(tile_half)
            diffTile = pto.alloc_tile(tile_half)
            xTile    = pto.alloc_tile(tile_row)

            with pto.if_context(row_offset_this_core < crow):
                rows_end_this_core = row_offset_this_core + rows_per_core
                need_truncate  = rows_end_this_core > crow
                remaining_rows = crow - row_offset_this_core
                rows_to_process = pto.select(need_truncate, remaining_rows, rows_per_core)

                for i in pto.for_range(c0, rows_to_process, c1):
                    row_idx  = i + row_offset_this_core
                    row_base = row_idx * ccol

                    for t in pto.for_range(c0, tiles_per_row, c1):
                        tile_base = row_base + t * ctile

                        sv_src_tile = pto.slice_view(subtensor_tile, source=tv_src, offsets=[tile_base],         sizes=[ctile])
                        sv_out_tile = pto.slice_view(subtensor_tile, source=tv_out, offsets=[tile_base],         sizes=[ctile])
                        sv_out_lo   = pto.slice_view(subtensor_half, source=tv_out, offsets=[tile_base],         sizes=[chalf])
                        sv_out_hi   = pto.slice_view(subtensor_half, source=tv_out, offsets=[tile_base + chalf], sizes=[chalf])

                        pto.load(sv_src_tile, xTile)

                        for stage in pto.for_range(c0, clog, c1):
                            pto.gather(xTile, evenTile, mask_pattern="P0101")
                            pto.gather(xTile, oddTile,  mask_pattern="P1010")
                            pto.add(evenTile, oddTile, sumTile)
                            pto.sub(evenTile, oddTile, diffTile)
                            pto.store(sumTile,  sv_out_lo)
                            pto.store(diffTile, sv_out_hi)
                            pto.load(sv_out_tile, xTile)

    _kernel.__name__ = fn_name
    return to_ir_module(meta_data=_meta_data)(_kernel)


if __name__ == "__main__":
    print(build_hadamard_kernel_dynamic(dtype="float32"))
