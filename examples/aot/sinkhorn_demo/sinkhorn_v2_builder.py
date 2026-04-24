"""
PTO-DSL port of ``kernel_sinkhorn_v2.cpp`` (K=4, fp16).

**Large ``N`` (``>= 2048``):** batched fast path — up to **32** matrices per
chunk (``128×16`` UB tile, same row packing as ``sinkhorn_batch8_builder.py``
but ``BATCH=32``). One bulk load/store per chunk, batched row softmax on the
``chunk_rows×K`` stack, then a short ``pto.range`` over matrices for
``tile.col_sum`` + ``eps`` + divide. This avoids the hand-interleaved layout
that must match ``tile.reshape`` byte-for-byte (a mismatch there produced
NaNs); it still cuts GM traffic and row-op fusion versus the small-``N`` path.

**Small ``N``:** same as ``sinkhorn_k4_builder.py`` (per-matrix, ``eps``).

Static ``repeat=10`` (host must pass 10).
"""

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

K = 4
TILE_COLS = 16
BATCH = 32
MAX_BATCH_ROWS = BATCH * K  # 128
CHUNK_MATRICES = BATCH

repeat = 10
N_FAST_THRESHOLD = 2048


def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    i32 = pto.int32
    ptr_fp16 = pto.PtrType(fp16)
    tensor2_fp16 = pto.TensorType(rank=2, dtype=fp16)
    sub_stack_rk_fp16 = pto.SubTensorType(shape=[MAX_BATCH_ROWS, K], dtype=fp16)

    row_cfg = pto.TileBufConfig()
    col_cfg = pto.TileBufConfig(blayout="ColMajor")

    stack_fp16 = pto.TileBufType(
        shape=[MAX_BATCH_ROWS, TILE_COLS],
        valid_shape=[-1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=row_cfg,
    )
    col_vec_stack_fp16 = pto.TileBufType(
        shape=[MAX_BATCH_ROWS, 1],
        valid_shape=[-1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=col_cfg,
    )
    row_vec_k_fp16 = pto.TileBufType(
        shape=[1, TILE_COLS],
        valid_shape=[-1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=row_cfg,
    )

    matrix_kk_fp16 = pto.TileBufType(
        shape=[TILE_COLS, TILE_COLS],
        valid_shape=[TILE_COLS, TILE_COLS],
        dtype=fp16,
        memory_space="VEC",
        config=row_cfg,
    )
    col_vec_k_fp16 = pto.TileBufType(
        shape=[TILE_COLS, 1],
        valid_shape=[-1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=col_cfg,
    )
    sub_kk_fp16 = pto.SubTensorType(shape=[K, K], dtype=fp16)

    return locals()


@to_ir_module(meta_data=meta_data)
def sinkhorn_v2_fp16(
    input_ptr: "ptr_fp16",
    output_ptr: "ptr_fp16",
    num_matrices_i32: "i32",
    repeat_i32: "i32",
    eps: "fp32",
) -> None:
    c0 = const(0)
    c1 = const(1)
    cK = const(K)
    c2048 = const(N_FAST_THRESHOLD)
    f0 = const(0.0, s.float32)
    f0_h = s.truncf(f0, s.float16)

    nm = s.index_cast(num_matrices_i32)
    eps_h = s.truncf(eps, s.float16)

    with pto.vector_section():
        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        num_blocks = pto.get_block_num()

        wid = s.index_cast(cid * sub_bnum + sub_bid)
        num_workers = s.index_cast(num_blocks * sub_bnum)

        n_rows = nm * cK

        tv_in = pto.as_tensor(
            tensor2_fp16,
            ptr=input_ptr,
            shape=[n_rows, cK],
            strides=[cK, c1],
        )
        tv_out = pto.as_tensor(
            tensor2_fp16,
            ptr=output_ptr,
            shape=[n_rows, cK],
            strides=[cK, c1],
        )

        with pto.if_context(s.ge(nm, c2048), has_else=True) as branch:

            base_pw = s.div_s(nm, num_workers)
            rem = s.rem_s(nm, num_workers)
            my_first = wid * base_pw + s.select(s.lt(wid, rem), wid, rem)
            my_count = base_pw + s.select(s.lt(wid, rem), c1, c0)

            cCHUNK = const(CHUNK_MATRICES)

            for chunk_off in pto.range(c0, my_count, cCHUNK):
                chunk_left = my_count - chunk_off
                chunk_mat = s.min_u(cCHUNK, chunk_left)
                chunk_rows = chunk_mat * cK

                row0 = my_first * cK + chunk_off * cK

                gm_in = pto.slice_view(
                    sub_stack_rk_fp16,
                    source=tv_in,
                    offsets=[row0, c0],
                    sizes=[chunk_rows, cK],
                )
                gm_out = pto.slice_view(
                    sub_stack_rk_fp16,
                    source=tv_out,
                    offsets=[row0, c0],
                    sizes=[chunk_rows, cK],
                )

                mat_batch = pto.alloc_tile(
                    stack_fp16, valid_row=chunk_rows, valid_col=cK
                )
                scratch_batch = pto.alloc_tile(
                    stack_fp16, valid_row=chunk_rows, valid_col=cK
                )
                row_stat = pto.alloc_tile(
                    col_vec_stack_fp16, valid_row=chunk_rows, valid_col=c1
                )
                col_stat = pto.alloc_tile(
                    row_vec_k_fp16, valid_row=c1, valid_col=cK
                )

                mat_wide = tile.subview(
                    mat_batch, [c0, c0], [MAX_BATCH_ROWS, TILE_COLS]
                )
                tile.muls(mat_wide, f0_h, mat_wide)
                mat_rk = tile.subview(mat_batch, [c0, c0], [MAX_BATCH_ROWS, K])
                pto.load(gm_in, mat_rk)

                tile.row_max(mat_batch, scratch_batch, row_stat)
                tile.row_expand_sub(mat_batch, row_stat, mat_batch)
                tile.exp(mat_batch, mat_batch)
                tile.row_sum(mat_batch, scratch_batch, row_stat)
                tile.row_expand_div(mat_batch, row_stat, mat_batch)

                mat_eps = tile.subview(
                    mat_batch, [c0, c0], [MAX_BATCH_ROWS, TILE_COLS]
                )
                tile.adds(mat_eps, eps_h, mat_eps)

                for m in pto.range(c0, chunk_mat, c1):
                    m_row = m * cK
                    mat_m = tile.subview(mat_batch, [m_row, c0], [K, K])
                    scratch_m = tile.subview(
                        scratch_batch, [m_row, c0], [K, K]
                    )
                    tile.col_sum(mat_m, scratch_m, col_stat, is_binary=True)
                    tile.adds(col_stat, eps_h, col_stat)
                    tile.col_expand_div(mat_m, col_stat, mat_m)

                for _ in range(1, repeat):
                    tile.row_sum(mat_batch, scratch_batch, row_stat)
                    tile.adds(row_stat, eps_h, row_stat)
                    tile.row_expand_div(mat_batch, row_stat, mat_batch)

                    for m in pto.range(c0, chunk_mat, c1):
                        m_row = m * cK
                        mat_m = tile.subview(mat_batch, [m_row, c0], [K, K])
                        scratch_m = tile.subview(
                            scratch_batch, [m_row, c0], [K, K]
                        )
                        tile.col_sum(
                            mat_m, scratch_m, col_stat, is_binary=True
                        )
                        tile.adds(col_stat, eps_h, col_stat)
                        tile.col_expand_div(mat_m, col_stat, mat_m)

                pto.store(mat_rk, gm_out)

        with branch.else_context():
            mat_full = pto.alloc_tile(matrix_kk_fp16)
            scratch_full = pto.alloc_tile(matrix_kk_fp16)
            row_stat_s = pto.alloc_tile(
                col_vec_k_fp16, valid_row=cK, valid_col=c1
            )
            col_stat_s = pto.alloc_tile(
                row_vec_k_fp16, valid_row=c1, valid_col=cK
            )
            mat_kk = tile.subview(mat_full, [c0, c0], [K, K])
            mat_eps_rows = tile.subview(mat_full, [c0, c0], [K, TILE_COLS])
            scratch_kk = tile.subview(scratch_full, [c0, c0], [K, K])

            for mi in pto.range(wid, nm, num_workers):
                row0 = mi * cK
                gm_in_s = pto.slice_view(
                    sub_kk_fp16,
                    source=tv_in,
                    offsets=[row0, c0],
                    sizes=[cK, cK],
                )
                gm_out_s = pto.slice_view(
                    sub_kk_fp16,
                    source=tv_out,
                    offsets=[row0, c0],
                    sizes=[cK, cK],
                )
                tile.muls(mat_full, f0_h, mat_full)
                pto.load(gm_in_s, mat_kk)
                tile.row_max(mat_kk, scratch_kk, row_stat_s)
                tile.row_expand_sub(mat_kk, row_stat_s, mat_kk)
                tile.exp(mat_kk, mat_kk)
                tile.row_sum(mat_kk, scratch_kk, row_stat_s)
                tile.row_expand_div(mat_kk, row_stat_s, mat_kk)
                tile.adds(mat_eps_rows, eps_h, mat_eps_rows)
                tile.col_sum(mat_kk, scratch_kk, col_stat_s, is_binary=True)
                tile.adds(col_stat_s, eps_h, col_stat_s)
                tile.col_expand_div(mat_kk, col_stat_s, mat_kk)
                for _ in range(1, repeat):
                    tile.row_sum(mat_kk, scratch_kk, row_stat_s)
                    tile.adds(row_stat_s, eps_h, row_stat_s)
                    tile.row_expand_div(mat_kk, row_stat_s, mat_kk)
                    tile.col_sum(
                        mat_kk, scratch_kk, col_stat_s, is_binary=True
                    )
                    tile.adds(col_stat_s, eps_h, col_stat_s)
                    tile.col_expand_div(mat_kk, col_stat_s, mat_kk)
                pto.store(mat_kk, gm_out_s)


if __name__ == "__main__":
    print(sinkhorn_v2_fp16)
