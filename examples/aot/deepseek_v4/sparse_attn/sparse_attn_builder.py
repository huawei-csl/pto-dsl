"""PTO DSL port of TileLang ``sparse_attn`` — full FlashAttention with
indexed (top-k) KV gather.

GPU semantics (per query (b, m)):
    For K top-k indices into ``kv[b, :, :]``, compute ``softmax(qK^T * scale
    || sink) @ V`` (sink is per-head additive logit; included in the
    softmax denominator but **dropped** from the V mix).

NPU implementation
------------------
Pure ``vector_section`` kernel. The matmul shapes are tiny (``[H, D] @
[D]`` per K-position) so we avoid multifunc cube↔vector pipes and
compute QK / PV incrementally per top-k position via VEC-pipe
``col_expand_mul`` + ``row_sum`` (an outer-product equivalent of matmul).

Per-head softmax state is **stored as full ``[H, D]`` tiles**
(replicated across the D axis) rather than as ``[H, 1]`` reductions —
this dodges the col-major⇄row-major reshape aliasing that
auto-sync analysis can miss. The replicated form is cheap on the
Ascend vector pipe (one ``row_expand`` to broadcast the per-head
``[H, 1]`` reduction back to ``[H, D]``) and lets every elementwise
softmax op operate on plain row-major tiles.

KV gather: each of the K positions is loaded individually from GM by

    1) ``pto.load_scalar(int32, idx_ptr, off)``   — read the index
    2) ``pto.slice_view`` with that dynamic row offset
    3) ``pto.load`` of one ``[1, D]`` row.

Online streaming softmax with `is_first` initialisation: on iter k=0
we set ``m_prev = logit``, ``l_run = 1``, ``acc_o = bcast(kv_row)``
directly; later iterations use the exp-rescaled update.

Tile shapes (H = H_PAD = 16, D = 128, all VEC, fp32 unless noted):

    q_tile        [H, D] fp16   — loaded once per query
    q_fp32        [H, D]
    kv_row_fp16   [1, D] fp16   — one position at a time
    kv_row_fp32   [1, D]
    kv_row_HD     [H, D]        — kv_row broadcast across heads
    tmp_HD        [H, D]        — q*kv_row scratch / outer scratch
    acc_o         [H, D]        — running output
    out_tile      [H, D] fp16
    logit_col     [H, 1] col    — per-head dot-product result
    red_tmp       [H, 1] col    — row_sum scratch
    sink_col      [H, 1] col    — attn_sink loaded once

    m_prev_HD     [H, D]   ┐
    m_new_HD      [H, D]   │  per-head softmax stats, replicated
    exp_diff_HD   [H, D]   │  across the D axis (m_prev_HD[h, d]
    p_HD          [H, D]   │  is the same scalar for every d).
    l_run_HD      [H, D]   ┘
"""

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

H_PAD = 16
D = 128
BLOCK = 64  # exported for util/test; no internal tiling effect


def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    i32 = pto.int32

    ptr_fp16 = pto.PtrType(fp16)
    ptr_fp32 = pto.PtrType(fp32)
    ptr_i32 = pto.PtrType(i32)

    tv_fp16_2d = pto.TensorType(rank=2, dtype=fp16)
    tv_fp32_2d = pto.TensorType(rank=2, dtype=fp32)

    sv_qrow = pto.SubTensorType(shape=[H_PAD, D], dtype=fp16)
    sv_kvrow = pto.SubTensorType(shape=[1, D], dtype=fp16)
    sv_sink_col = pto.SubTensorType(shape=[H_PAD, 1], dtype=fp32)
    sv_orow = pto.SubTensorType(shape=[H_PAD, D], dtype=fp16)

    row_cfg = pto.TileBufConfig()
    # row_sum dst must be col-major [H, 1] (NoneBox). row_expand_mul
    # accepts col-major src1 (verified in fa_builder). We avoid plain
    # `row_expand` (which needs row-major src) by broadcasting via
    # row_expand_mul with a constant-ones tile.
    col_cfg = pto.TileBufConfig(blayout="ColMajor", slayout="NoneBox")

    tile_q_fp16 = pto.TileBufType(
        shape=[H_PAD, D], dtype=fp16, memory_space="VEC", config=row_cfg
    )
    tile_HD_fp32 = pto.TileBufType(
        shape=[H_PAD, D], dtype=fp32, memory_space="VEC", config=row_cfg
    )
    tile_o_fp16 = pto.TileBufType(
        shape=[H_PAD, D], dtype=fp16, memory_space="VEC", config=row_cfg
    )
    tile_kv_fp16 = pto.TileBufType(
        shape=[1, D], dtype=fp16, memory_space="VEC", config=row_cfg
    )
    tile_kv_fp32 = pto.TileBufType(
        shape=[1, D], dtype=fp32, memory_space="VEC", config=row_cfg
    )
    tile_col_stat = pto.TileBufType(
        shape=[H_PAD, 1],
        dtype=fp32,
        memory_space="VEC",
        config=col_cfg,
    )
    return locals()


@to_ir_module(meta_data=meta_data)
def sparse_attn(
    q_ptr: "ptr_fp16",
    kv_ptr: "ptr_fp16",
    o_ptr: "ptr_fp16",
    sink_ptr: "ptr_fp32",
    idx_ptr: "ptr_i32",
    B_i32: "i32",
    M_i32: "i32",
    N_i32: "i32",
    TOPK_i32: "i32",
    scale_f32: "fp32",
) -> None:
    c0 = const(0)
    c1 = const(1)
    cH = const(H_PAD)
    cD = const(D)

    f0 = const(0.0, s.float32)
    f1 = const(1.0, s.float32)

    B = s.index_cast(B_i32)
    M = s.index_cast(M_i32)
    N = s.index_cast(N_i32)
    TOPK = s.index_cast(TOPK_i32)

    with pto.vector_section():
        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        num_blocks = pto.get_block_num()
        wid = s.index_cast(cid * sub_bnum + sub_bid)
        ncores = s.index_cast(num_blocks * sub_bnum)

        total = B * M

        # --- GM tensor views ----------------------------------------
        tvQ = pto.as_tensor(
            tv_fp16_2d, ptr=q_ptr, shape=[total * cH, cD], strides=[cD, c1]
        )
        tvKV = pto.as_tensor(
            tv_fp16_2d, ptr=kv_ptr, shape=[B * N, cD], strides=[cD, c1]
        )
        tvO = pto.as_tensor(
            tv_fp16_2d, ptr=o_ptr, shape=[total * cH, cD], strides=[cD, c1]
        )
        # Sink fp32 [H_PAD] viewed as [H_PAD, 1] (1D contiguous == [H, 1]
        # col-stride-1 == [H, 1] row-stride-1).
        tv_sink = pto.as_tensor(
            tv_fp32_2d, ptr=sink_ptr, shape=[cH, c1], strides=[c1, c1]
        )

        # --- Tile buffers -------------------------------------------
        q_tile = pto.alloc_tile(tile_q_fp16)
        q_fp32 = pto.alloc_tile(tile_HD_fp32)
        kv_row_fp16 = pto.alloc_tile(tile_kv_fp16)
        kv_row_fp32 = pto.alloc_tile(tile_kv_fp32)
        kv_row_HD = pto.alloc_tile(tile_HD_fp32)

        tmp_HD = pto.alloc_tile(tile_HD_fp32)
        acc_o = pto.alloc_tile(tile_HD_fp32)
        # All-ones [H, D] used to broadcast [H, 1] col-major reductions
        # via row_expand_mul (which accepts col-major src1, unlike plain
        # row_expand which requires row-major src).
        ones_HD = pto.alloc_tile(tile_HD_fp32)

        # Per-head softmax stats, replicated across D so all elementwise
        # ops stay on row-major [H, D] tiles.
        m_prev_HD = pto.alloc_tile(tile_HD_fp32)
        m_new_HD = pto.alloc_tile(tile_HD_fp32)
        exp_diff_HD = pto.alloc_tile(tile_HD_fp32)
        p_HD = pto.alloc_tile(tile_HD_fp32)
        l_run_HD = pto.alloc_tile(tile_HD_fp32)

        logit_col = pto.alloc_tile(tile_col_stat)
        # row_sum tmp must be same shape/layout as the src tile.
        red_tmp = pto.alloc_tile(tile_HD_fp32)
        sink_col = pto.alloc_tile(tile_col_stat)

        out_tile = pto.alloc_tile(tile_o_fp16)

        # --- Sink load (once per worker) ----------------------------
        sink_view = pto.slice_view(
            sv_sink_col, source=tv_sink, offsets=[c0, c0], sizes=[cH, c1]
        )
        pto.load(sink_view, sink_col)

        # --- Per-query loop -----------------------------------------
        for bm in pto.range(wid, total, ncores):
            b = bm // M

            # Load Q[bm*H : bm*H + H, :] in fp16, then cvt to fp32.
            q_off = bm * cH
            q_view = pto.slice_view(
                sv_qrow, source=tvQ, offsets=[q_off, c0], sizes=[cH, cD]
            )
            pto.load(q_view, q_tile)
            tile.cvt(q_tile, q_fp32)
            # Build ones_HD from q_fp32 (guaranteed finite – randn fp16
            # has no NaN/Inf), so `q_fp32 * 0 + 1 = 1` everywhere.
            tile.muls(q_fp32, f0, ones_HD)
            tile.adds(ones_HD, f1, ones_HD)

            idx_base = bm * TOPK
            kv_base = b * N

            for k in pto.range(c0, TOPK, c1):
                is_first = s.eq(k, c0)

                # ---- Gather one KV row by index --------------------
                idx_off = idx_base + k
                idx_i32 = pto.load_scalar(s.int32, idx_ptr, idx_off)
                idx_idx = s.index_cast(idx_i32)
                kv_row_off = kv_base + idx_idx
                kv_view = pto.slice_view(
                    sv_kvrow,
                    source=tvKV,
                    offsets=[kv_row_off, c0],
                    sizes=[c1, cD],
                )
                pto.load(kv_view, kv_row_fp16)
                tile.cvt(kv_row_fp16, kv_row_fp32)

                # Broadcast kv_row [1, D] across heads → kv_row_HD [H, D].
                tile.col_expand_mul(ones_HD, kv_row_fp32, kv_row_HD)

                # ---- QK: logit_col[h] = (q · kv_row)[h] * scale ----
                tile.col_expand_mul(q_fp32, kv_row_fp32, tmp_HD)
                tile.row_sum(tmp_HD, red_tmp, logit_col)
                # Broadcast logit_col [H, 1] col-major → [H, D] row-major
                # via row_expand_mul against the ones tile.
                tile.row_expand_mul(ones_HD, logit_col, tmp_HD)
                tile.muls(tmp_HD, scale_f32, tmp_HD)
                # tmp_HD now holds logit replicated across D.

                with pto.if_context(is_first, has_else=True) as br:
                    # First position: m_prev = logit; l_run = 1;
                    # acc_o[h, d] = kv_row[d] (broadcast).
                    tile.muls(tmp_HD, f1, m_prev_HD)
                    tile.muls(tmp_HD, f0, l_run_HD)
                    tile.adds(l_run_HD, f1, l_run_HD)
                    tile.muls(kv_row_HD, f1, acc_o)

                with br.else_context():
                    # m_new = max(m_prev, logit)   (per-head, replicated)
                    tile.max(m_prev_HD, tmp_HD, m_new_HD)
                    # exp_diff = exp(m_prev - m_new)
                    tile.sub(m_prev_HD, m_new_HD, exp_diff_HD)
                    tile.exp(exp_diff_HD, exp_diff_HD)
                    # p = exp(logit - m_new)
                    tile.sub(tmp_HD, m_new_HD, p_HD)
                    tile.exp(p_HD, p_HD)
                    # l_run = exp_diff * l_run + p
                    tile.mul(l_run_HD, exp_diff_HD, l_run_HD)
                    tile.add(l_run_HD, p_HD, l_run_HD)
                    # acc_o = exp_diff * acc_o + p * kv_row_HD
                    tile.mul(acc_o, exp_diff_HD, acc_o)
                    tile.mul(p_HD, kv_row_HD, tmp_HD)
                    tile.add(acc_o, tmp_HD, acc_o)
                    # m_prev = m_new
                    tile.muls(m_new_HD, f1, m_prev_HD)

            # --- Finalise: l_run += exp(sink - m_prev); acc_o /= l_run.
            # Broadcast sink_col [H, 1] → tmp_HD [H, D] via row_expand_mul.
            tile.row_expand_mul(ones_HD, sink_col, tmp_HD)
            tile.sub(tmp_HD, m_prev_HD, exp_diff_HD)
            tile.exp(exp_diff_HD, exp_diff_HD)
            tile.add(l_run_HD, exp_diff_HD, l_run_HD)
            tile.div(acc_o, l_run_HD, acc_o)

            tile.cvt(acc_o, out_tile)
            o_view = pto.slice_view(
                sv_orow, source=tvO, offsets=[q_off, c0], sizes=[cH, cD]
            )
            pto.store(out_tile, o_view)


if __name__ == "__main__":
    print(sparse_attn)
