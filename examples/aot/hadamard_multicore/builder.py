from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const

_DTYPE_MAP = {
    "float32": lambda: pto.float32,
    "float16": lambda: pto.float16,
}


def meta_data(src_dtype=None, rows=32, cols=32):
    if src_dtype is None:
        src_dtype = pto.float32
    if isinstance(src_dtype, str):
        src_dtype = _DTYPE_MAP[src_dtype]()

    assert cols % 2 == 0, "FWHT requires cols (n) to be even."
    half = cols // 2

    tile_cfg = pto.TileBufConfig()

    return {
        "ptr_src": pto.PtrType(src_dtype),
        "ptr_out": pto.PtrType(src_dtype),

        "tv2_src": pto.TensorType(rank=2, dtype=src_dtype),
        "tv2_out": pto.TensorType(rank=2, dtype=src_dtype),

        # GM slice views (row-wise)
        "tile_view_row":  pto.SubTensorType(shape=[1, cols], dtype=src_dtype),
        "tile_view_half": pto.SubTensorType(shape=[1, half], dtype=src_dtype),

        # UB tiles (row-wise)
        "tile_buf_row": pto.TileBufType(
            shape=[1, cols], valid_shape=[1, cols],
            dtype=src_dtype, memory_space="VEC", config=tile_cfg
        ),
        "tile_buf_half": pto.TileBufType(
            shape=[1, half], valid_shape=[1, half],
            dtype=src_dtype, memory_space="VEC", config=tile_cfg
        ),
    }

def build_hadamard_kernel(fn_name="hadamard_kernel", dtype=None, rows=32, cols=32):
    assert cols > 0 and (cols & (cols - 1) == 0), "cols (n) must be a power of two."
    log2_n = cols.bit_length() - 1
    half = cols // 2

    _meta_data = lambda: meta_data(dtype, rows, cols)

    def _kernel(
        arg0: "ptr_src",
        arg1: "ptr_out",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        crow = const(rows)
        ccol = const(cols)
        chalf = const(half)
        clog = const(log2_n)

        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        cidmul = cid * sub_bnum
        vid = cidmul + sub_bid
        num_blocks = pto.get_block_num()

        vid_idx = pto.index_cast(vid)
        num_cores = pto.index_cast(num_blocks)

        rows_per_core = pto.ceil_div(crow, num_cores)
        row_start = vid_idx * rows_per_core
        row_end_unclamped = row_start + rows_per_core

        with pto.vector_section():
            tv_src = pto.as_tensor(tv2_src, ptr=arg0, shape=[crow, ccol], strides=[ccol, c1])
            tv_out = pto.as_tensor(tv2_out, ptr=arg1, shape=[crow, ccol], strides=[ccol, c1])

            evenTile = pto.alloc_tile(tile_buf_half)
            oddTile  = pto.alloc_tile(tile_buf_half)
            sumTile  = pto.alloc_tile(tile_buf_half)
            diffTile = pto.alloc_tile(tile_buf_half)
            xTile    = pto.alloc_tile(tile_buf_row)

            with pto.if_context(row_start < crow):
                need_trunc = row_end_unclamped > crow
                row_end = pto.select(need_trunc, crow, row_end_unclamped)

                for s in pto.for_range(row_start, row_end, c1):
                    sv_src_row = pto.slice_view(tile_view_row,  source=tv_src, offsets=[s, c0],    sizes=[c1, ccol])
                    sv_out_row = pto.slice_view(tile_view_row,  source=tv_out, offsets=[s, c0],    sizes=[c1, ccol])
                    sv_out_lo  = pto.slice_view(tile_view_half, source=tv_out, offsets=[s, c0],    sizes=[c1, chalf])
                    sv_out_hi  = pto.slice_view(tile_view_half, source=tv_out, offsets=[s, chalf], sizes=[c1, chalf])

                    for i in pto.for_range(c0, clog, c1):
                        with pto.if_context(pto.eq(i, c0)) as branch:
                            pto.load(sv_src_row, xTile)
                        with branch.else_():
                            pto.load(sv_out_row, xTile)
                        pto.gather(xTile, evenTile, mask_pattern="P0101")
                        pto.gather(xTile, oddTile,  mask_pattern="P1010")
                        pto.add(evenTile, oddTile, sumTile)
                        pto.sub(evenTile, oddTile, diffTile)
                        pto.store(sumTile,  sv_out_lo)
                        pto.store(diffTile, sv_out_hi)
                    

               

    _kernel.__name__ = fn_name
    return to_ir_module(meta_data=_meta_data)(_kernel)