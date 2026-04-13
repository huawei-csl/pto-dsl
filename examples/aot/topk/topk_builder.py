"""
TopK AOT kernel: for each row of an [N_ROWS × N_COLS] float32 matrix, find the
top-TOPK elements and return their values and original column indices.

Pipeline (per row)
------------------
  1. TSORT32   – sort within SORT_BLOCK_LEN-element blocks, writing interleaved
                 (score_f32, idx_u32) pairs to the sort buffer.
  2. TMRGSORT  – multi-pass 4-way merge until the sort buffer is fully sorted
                 descending by score. Unrolled at builder time (static sizes).
  3. TMOV tb_sort → tb_gather_win (valid_shape=[1, 2*TOPK]).
  4. TGATHER P0101 on tb_gather_win – extract top-TOPK scores (even slots).
  5. TGATHER P1010 on tb_gather_win – extract top-TOPK indices (odd slots,
                 stored as uint32 bit-patterns in a float32 tile).
  6. TSTORE    – write scores and indices to global memory.

The gather-window tile has the same physical shape as the sort buffer but its
valid_shape is limited to [1, 2*TOPK].  This ensures TGATHER P0101/P1010 sees
exactly 2*TOPK elements and produces exactly TOPK outputs, even when TOPK < N_COLS
(without the window, P0101 on a sort_cols-element tile would produce N_COLS
outputs and overflow the TOPK-element destination tile).

Constraints (verified by assertions in build_topk)
---------------------------------------------------
  * TOPK must be ≤ N_COLS.
  * N_ROWS is unconstrained at compile time (any value works at runtime).
  * HW_BLOCK_LEN (= SORT_BLOCK_LEN × DST_STRIDE) must be a multiple of 64.
  * SORT_COLS (= N_COLS × DST_STRIDE) must be an exact power-of-4 multiple of
    HW_BLOCK_LEN (guarantees a clean merge with no tail block).

Valid N_COLS values (SORT_BLOCK_LEN=32)
---------------------------------------
  SORT_COLS = N_COLS*2 must be a power-of-4 multiple of HW_BLOCK_LEN=64:
    N_COLS =  128 → SORT_COLS =  256  (1 merge pass)
    N_COLS =  512 → SORT_COLS = 1024  (2 merge passes)
    N_COLS = 2048 → SORT_COLS = 4096  (3 merge passes)

Usage
-----
  python topk_builder.py                         # default: n_cols=512 topk=256
  python topk_builder.py --n-cols 128 --topk 64
"""

import argparse

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

# float32: TSORT32 expands each input element to (score_f32, idx_u32) = 2 words.
_DST_STRIDE = 2
_SORT_BLOCK_LEN = 32  # TSORT32 sorts within blocks of this many input elements


def fn_name(n_cols: int, topk: int) -> str:
    """Unique kernel name (n_rows is dynamic and not encoded)."""
    return f"topk_c{n_cols}_k{topk}"


def build_topk(
    n_cols: int = 512,
    topk: int = 256,
    sort_block_len: int = _SORT_BLOCK_LEN,
):
    """Return a compiled MLIR module for the given compile-time TopK shape.

    n_rows is NOT a compile-time parameter – it is a runtime ``int32`` argument
    (``argN``) passed at each invocation.  The kernel uses ``s.ceil_div`` to
    distribute rows across blocks and guards the last block with ``if_context``,
    so any n_rows value is supported without recompilation.
    """
    sort_cols = n_cols * _DST_STRIDE
    hw_block_len = sort_block_len * _DST_STRIDE

    assert topk <= n_cols, f"topk={topk} must be ≤ n_cols={n_cols}"
    assert (
        hw_block_len % 64 == 0
    ), f"hw_block_len={hw_block_len} must be a multiple of 64"
    _blk = hw_block_len
    while _blk * 4 <= sort_cols:
        _blk *= 4
    assert _blk == sort_cols, (
        f"sort_cols={sort_cols} is not a power-of-4 multiple of hw_block_len={hw_block_len}; "
        "tail merging is not implemented in this example."
    )

    def _meta_data():
        f32 = pto.float32
        u32 = pto.uint32
        tile_cfg = pto.TileBufConfig()
        return {
            "ptr_f32": pto.PtrType(f32),
            "ptr_u32": pto.PtrType(u32),
            "index_dtype": pto.int32,
            "tensor_src": pto.TensorType(rank=2, dtype=f32),
            "tensor_inidx": pto.TensorType(rank=2, dtype=u32),
            "tensor_scores": pto.TensorType(rank=2, dtype=f32),
            "tensor_indices": pto.TensorType(rank=2, dtype=u32),
            "sub_src": pto.SubTensorType(shape=[1, n_cols], dtype=f32),
            "sub_inidx": pto.SubTensorType(shape=[1, n_cols], dtype=u32),
            "sub_scores": pto.SubTensorType(shape=[1, topk], dtype=f32),
            "sub_indices": pto.SubTensorType(shape=[1, topk], dtype=u32),
            "tile_src": pto.TileBufType(
                shape=[1, n_cols],
                valid_shape=[1, n_cols],
                dtype=f32,
                memory_space="VEC",
                config=tile_cfg,
            ),
            "tile_inidx": pto.TileBufType(
                shape=[1, n_cols],
                valid_shape=[1, n_cols],
                dtype=u32,
                memory_space="VEC",
                config=tile_cfg,
            ),
            "tile_sort_f32": pto.TileBufType(
                shape=[1, sort_cols],
                valid_shape=[1, sort_cols],
                dtype=f32,
                memory_space="VEC",
                config=tile_cfg,
            ),
            "tile_sort_u32": pto.TileBufType(
                shape=[1, sort_cols],
                valid_shape=[1, sort_cols],
                dtype=u32,
                memory_space="VEC",
                config=tile_cfg,
            ),
            # Gather window: same physical shape as tile_sort, but valid_shape
            # limited to [1, 2*topk] so TGATHER P0101/P1010 produces topk outputs.
            "tile_gather_win_f32": pto.TileBufType(
                shape=[1, sort_cols],
                valid_shape=[1, 2 * topk],
                dtype=f32,
                memory_space="VEC",
                config=tile_cfg,
            ),
            "tile_gather_win_u32": pto.TileBufType(
                shape=[1, sort_cols],
                valid_shape=[1, 2 * topk],
                dtype=u32,
                memory_space="VEC",
                config=tile_cfg,
            ),
            "tile_topk_f32": pto.TileBufType(
                shape=[1, topk],
                valid_shape=[1, topk],
                dtype=f32,
                memory_space="VEC",
                config=tile_cfg,
            ),
            "tile_topk_u32": pto.TileBufType(
                shape=[1, topk],
                valid_shape=[1, topk],
                dtype=u32,
                memory_space="VEC",
                config=tile_cfg,
            ),
        }

    def _kernel(
        src_ptr: "ptr_f32",  # [n_rows, n_cols]  float32 – input scores
        inidx_ptr: "ptr_u32",  # [n_cols]          uint32  – original column indices
        scores_ptr: "ptr_f32",  # [n_rows, topk]    float32 – output top-k scores
        indices_ptr: "ptr_u32",  # [n_rows, topk]    uint32  – output top-k indices
        argN: "index_dtype",  # n_rows (runtime)
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_ncols = const(n_cols)
        c_topk = const(topk)
        n_rows_dyn = s.index_cast(argN)

        with pto.vector_section():
            cid = pto.get_block_idx()
            sub_bid = pto.get_subblock_idx()
            sub_bnum = pto.get_subblock_num()
            vid = s.index_cast(cid * sub_bnum + sub_bid)
            n_blocks = s.index_cast(pto.get_block_num() * sub_bnum)

            # Distribute rows across logical cores (cube_cores * sub_bnum) with ceil_div.
            rows_per_core = s.ceil_div(n_rows_dyn, n_blocks)
            row_start = vid * rows_per_core
            row_end_raw = row_start + rows_per_core
            need_clamp = row_end_raw > n_rows_dyn
            rows_this_core = s.select(need_clamp, n_rows_dyn - row_start, rows_per_core)
            tv_src = pto.as_tensor(
                tensor_src,
                ptr=src_ptr,
                shape=[n_rows_dyn, c_ncols],
                strides=[c_ncols, c1],
            )
            tv_inidx = pto.as_tensor(
                tensor_inidx, ptr=inidx_ptr, shape=[c1, c_ncols], strides=[c_ncols, c1]
            )
            tv_scores = pto.as_tensor(
                tensor_scores,
                ptr=scores_ptr,
                shape=[n_rows_dyn, c_topk],
                strides=[c_topk, c1],
            )
            tv_indices = pto.as_tensor(
                tensor_indices,
                ptr=indices_ptr,
                shape=[n_rows_dyn, c_topk],
                strides=[c_topk, c1],
            )

            tb_src = pto.alloc_tile(tile_src)
            tb_inidx = pto.alloc_tile(tile_inidx)
            tb_sort = pto.alloc_tile(tile_sort_f32)
            tb_sort_tmp = pto.alloc_tile(tile_sort_f32)
            tb_gather_win_f = pto.alloc_tile(tile_gather_win_f32)
            tb_gather_win_u = pto.alloc_tile(tile_gather_win_u32)
            tb_scores = pto.alloc_tile(tile_topk_f32)
            tb_indices = pto.alloc_tile(tile_topk_u32)

            # Load shared column-index vector once per core.
            sv_inidx = pto.slice_view(
                sub_inidx, source=tv_inidx, offsets=[c0, c0], sizes=[c1, c_ncols]
            )
            pto.load(sv_inidx, tb_inidx)

            # Guard: blocks beyond n_rows do nothing.
            with pto.if_context(row_start < n_rows_dyn):
                with pto.if_context(rows_this_core > c0):
                    for i in pto.range(c0, rows_this_core, c1):
                        row = i + row_start

                        # 1. Load input row.
                        sv_src = pto.slice_view(
                            sub_src,
                            source=tv_src,
                            offsets=[row, c0],
                            sizes=[c1, c_ncols],
                        )
                        pto.load(sv_src, tb_src)

                        # 2. TSORT32: sort within sort_block_len-element blocks.
                        tile.sort32(tb_src, tb_sort, tb_inidx)

                        # 3. Multi-pass TMRGSORT (unrolled at build time).
                        cur_block = hw_block_len
                        while cur_block * 4 <= sort_cols:
                            tile.mrgsort(tb_sort, tb_sort_tmp, const(cur_block))
                            tile.mov(tb_sort_tmp, tb_sort)
                            cur_block *= 4

                        # 4. Copy into gather window (valid_shape=[1, 2*topk]).
                        tile.mov(tb_sort, tb_gather_win_f)

                        # 5. Extract top-topk scores (even slots = score_f32).
                        tile.gather(tb_gather_win_f, tb_scores, mask_pattern="P0101")

                        # 6. Extract top-topk indices (odd slots = idx_u32 bits).
                        tile.mov(tb_sort, tb_gather_win_u)
                        tile.gather(tb_gather_win_u, tb_indices, mask_pattern="P1010")

                        # 7. Store outputs.
                        sv_scores = pto.slice_view(
                            sub_scores,
                            source=tv_scores,
                            offsets=[row, c0],
                            sizes=[c1, c_topk],
                        )
                        pto.store(tb_scores, sv_scores)

                        sv_indices = pto.slice_view(
                            sub_indices,
                            source=tv_indices,
                            offsets=[row, c0],
                            sizes=[c1, c_topk],
                        )
                        pto.store(tb_indices, sv_indices)

    _kernel.__name__ = fn_name(n_cols, topk)
    return to_ir_module(meta_data=_meta_data)(_kernel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print MLIR IR for a TopK kernel")
    parser.add_argument("--n-cols", type=int, default=512)
    parser.add_argument("--topk", type=int, default=256)
    args = parser.parse_args()
    print(build_topk(n_cols=args.n_cols, topk=args.topk))
