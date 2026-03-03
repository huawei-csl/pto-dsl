from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const

DTYPES = {
    "float32": lambda: pto.float32,
    "float16": lambda: pto.float16,
}


def meta_data(dtype=None, cols=32):
    if dtype is None:
        dtype = "float32"
    if isinstance(dtype, str):
        dtype = DTYPES[dtype]()

    half = cols // 2
    tile_cfg = pto.TileBufConfig()

    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=1, dtype=dtype)
    subtensor_row = pto.SubTensorType(shape=[1, cols], dtype=dtype)
    subtensor_half = pto.SubTensorType(shape=[1, half], dtype=dtype)
    tile_row = pto.TileBufType(
        shape=[1, cols], valid_shape=[1, cols],
        dtype=dtype, memory_space="VEC", config=tile_cfg,
    )
    tile_half = pto.TileBufType(
        shape=[1, half], valid_shape=[1, half],
        dtype=dtype, memory_space="VEC", config=tile_cfg,
    )

    return {
        "ptr_type": ptr_type,
        "index_dtype": pto.int32,
        "tensor_type": tensor_type,
        "subtensor_row": subtensor_row,
        "subtensor_half": subtensor_half,
        "tile_row": tile_row,
        "tile_half": tile_half,
    }


def build_hadamard_kernel(fn_name="hadamard_kernel", dtype=None, cols=32):
    assert cols > 0 and (cols & (cols - 1) == 0), "cols (n) must be a power of two."
    log2_n = cols.bit_length() - 1
    half = cols // 2

    _meta_data = lambda: meta_data(dtype, cols)

    def _kernel(
        arg0: "ptr_type",
        arg1: "ptr_type",
        argN: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        ccol = const(cols)
        chalf = const(half)
        clog = const(log2_n)

        crow = pto.index_cast(argN)
        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        vid = cid * sub_bnum + sub_bid
        num_blocks = pto.get_block_num()

        vid_idx = pto.index_cast(vid)
        num_cores = pto.index_cast(num_blocks)

        total_elements = crow * ccol
        rows_per_core = pto.ceil_div(crow, num_cores)
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
                need_truncate = rows_end_this_core > crow
                remaining_rows = crow - row_offset_this_core
                rows_to_process = pto.select(need_truncate, remaining_rows, rows_per_core)

                for i in pto.for_range(c0, rows_to_process, c1):
                    row_idx = i + row_offset_this_core
                    offset = row_idx * ccol

                    sv_src_row = pto.slice_view(subtensor_row,  source=tv_src, offsets=[offset],         sizes=[ccol])
                    sv_src_hi  = pto.slice_view(subtensor_half, source=tv_src, offsets=[offset + chalf], sizes=[chalf])
                    sv_out_row = pto.slice_view(subtensor_row,  source=tv_out, offsets=[offset],         sizes=[ccol])
                    sv_out_lo  = pto.slice_view(subtensor_half, source=tv_out, offsets=[offset],         sizes=[chalf])
                    sv_out_hi  = pto.slice_view(subtensor_half, source=tv_out, offsets=[offset + chalf], sizes=[chalf])

                    pto.load(sv_src_row, xTile)
                    pto.load(sv_src_hi, oddTile)
                    pto.barrier("TLOAD")

                    for stage in pto.for_range(c0, c1, c1):
                        pto.gather(xTile, evenTile, mask_pattern="P0101")
                        pto.barrier("TVEC")
                        #pto.gather(xTile, oddTile,  mask_pattern="P1010")
                        pto.add(evenTile, oddTile, sumTile)
                        pto.barrier("TVEC")
                        pto.sub(evenTile, oddTile, diffTile)
                        pto.barrier("TVEC")
                        pto.store(sumTile,  sv_out_lo)
                        pto.store(diffTile, sv_out_hi)
                        pto.barrier("TSTORE_VEC")
                        pto.load(sv_out_row, xTile)
                        pto.load(sv_out_hi, oddTile)
                        pto.barrier("TLOAD")

    _kernel.__name__ = fn_name
    return to_ir_module(meta_data=_meta_data)(_kernel)


if __name__ == "__main__":
    print(build_hadamard_kernel(dtype="float32", cols=32))
