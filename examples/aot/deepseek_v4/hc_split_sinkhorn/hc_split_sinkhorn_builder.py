"""PTO DSL port of TileLang ``hc_split_sinkhorn`` — full on-device kernel.

Original (GPU): given ``mixes[n, (2+hc)*hc]`` plus ``hc_scale[3]`` and
``hc_base[(2+hc)*hc]``, fuse three heads:

    pre[n, hc]      = sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post[n, hc]     = 2 * sigmoid(mixes[..., hc:2hc] * scale[1] + base[hc:2hc])
    comb[n, hc, hc] = sinkhorn-normalize((mixes[..., 2hc:] * scale[2]
                       + base[2hc:]).reshape(n, hc, hc))

NPU port: a single vector_section runs all three heads end-to-end.
Sigmoid is composed from ``tile.muls(-1) + tile.exp + tile.adds(1) +
tile.reciprocal``. The three scalar scales are read once via
``pto.load_scalar``; the three base tensors (pre/post/comb portions of
``hc_base``) are loaded into VEC tiles once per worker.

Shapes (HC = 4 → padded to TILE_DIM = 16 for op alignment):

    mixes:    [n, MIX_HC]   fp32     (MIX_HC = 24)
    hc_scale: [3]           fp32
    hc_base:  [MIX_HC]      fp32
    pre:      [n, HC]       fp32     (output)
    post:     [n, HC]       fp32     (output)
    comb:     [n, HC, HC]   fp32     (output)
"""

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

HC = 4
MIX_HC = (2 + HC) * HC  # 24
TILE_DIM = 16
SINKHORN_ITERS = 20
EPS = 1e-6


def meta_data():
    fp32 = pto.float32
    i32 = pto.int32
    ptr_fp32 = pto.PtrType(fp32)

    tv2 = pto.TensorType(rank=2, dtype=fp32)
    tv3 = pto.TensorType(rank=3, dtype=fp32)

    sv_row = pto.SubTensorType(shape=[1, HC], dtype=fp32)
    sv_kk = pto.SubTensorType(shape=[HC, HC], dtype=fp32)

    row_cfg = pto.TileBufConfig()
    col_cfg = pto.TileBufConfig(blayout="ColMajor")

    tile_full = pto.TileBufType(
        shape=[TILE_DIM, TILE_DIM], dtype=fp32, memory_space="VEC", config=row_cfg
    )
    tile_row_stat = pto.TileBufType(
        shape=[TILE_DIM, 1],
        valid_shape=[-1, -1],
        dtype=fp32,
        memory_space="VEC",
        config=col_cfg,
    )
    tile_col_stat = pto.TileBufType(
        shape=[1, TILE_DIM],
        valid_shape=[-1, -1],
        dtype=fp32,
        memory_space="VEC",
        config=row_cfg,
    )
    return locals()


@to_ir_module(meta_data=meta_data)
def hc_split_sinkhorn(
    mixes_ptr: "ptr_fp32",
    hc_scale_ptr: "ptr_fp32",
    hc_base_ptr: "ptr_fp32",
    pre_ptr: "ptr_fp32",
    post_ptr: "ptr_fp32",
    comb_ptr: "ptr_fp32",
    n_i32: "i32",
) -> None:
    c0 = const(0)
    c1 = const(1)
    c2 = const(2)
    cHC = const(HC)
    cMIX = const(MIX_HC)
    c2HC = const(2 * HC)
    cHCHC = const(HC * HC)

    eps = const(EPS, s.float32)
    one_f = const(1.0, s.float32)
    neg1_f = const(-1.0, s.float32)
    two_f = const(2.0, s.float32)

    n = s.index_cast(n_i32)

    with pto.vector_section():
        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        num_blocks = pto.get_block_num()
        wid = s.index_cast(cid * sub_bnum + sub_bid)
        ncores = s.index_cast(num_blocks * sub_bnum)

        # Pointer slicing for the three head regions of mixes / hc_base.
        mixes_post_ptr = pto.add_ptr(mixes_ptr, cHC)
        mixes_comb_ptr = pto.add_ptr(mixes_ptr, c2HC)
        base_post_ptr = pto.add_ptr(hc_base_ptr, cHC)
        base_comb_ptr = pto.add_ptr(hc_base_ptr, c2HC)

        # mixes views: pre/post are [n, HC] columns of [n, MIX_HC]
        # (row stride = MIX_HC); comb is [n, HC, HC] viewing the trailing
        # HC*HC contiguous block per sample.
        tv_mix_pre = pto.as_tensor(
            tv2, ptr=mixes_ptr, shape=[n, cHC], strides=[cMIX, c1]
        )
        tv_mix_post = pto.as_tensor(
            tv2, ptr=mixes_post_ptr, shape=[n, cHC], strides=[cMIX, c1]
        )
        tv_mix_comb = pto.as_tensor(
            tv3,
            ptr=mixes_comb_ptr,
            shape=[n, cHC, cHC],
            strides=[cMIX, cHC, c1],
        )

        # Base tensors (loaded once per worker, contiguous within hc_base).
        tv_base_pre = pto.as_tensor(
            tv2, ptr=hc_base_ptr, shape=[c1, cHC], strides=[cHC, c1]
        )
        tv_base_post = pto.as_tensor(
            tv2, ptr=base_post_ptr, shape=[c1, cHC], strides=[cHC, c1]
        )
        tv_base_comb = pto.as_tensor(
            tv2, ptr=base_comb_ptr, shape=[cHC, cHC], strides=[cHC, c1]
        )

        # Output tensors.
        tv_pre_out = pto.as_tensor(tv2, ptr=pre_ptr, shape=[n, cHC], strides=[cHC, c1])
        tv_post_out = pto.as_tensor(
            tv2, ptr=post_ptr, shape=[n, cHC], strides=[cHC, c1]
        )
        tv_comb_out = pto.as_tensor(
            tv3,
            ptr=comb_ptr,
            shape=[n, cHC, cHC],
            strides=[cHCHC, cHC, c1],
        )

        # Tile buffers: full TILE_DIM × TILE_DIM with valid sub-rectangles.
        pre_full = pto.alloc_tile(tile_full)
        post_full = pto.alloc_tile(tile_full)
        comb_full = pto.alloc_tile(tile_full)
        scratch_full = pto.alloc_tile(tile_full)
        base_pre_full = pto.alloc_tile(tile_full)
        base_post_full = pto.alloc_tile(tile_full)
        base_comb_full = pto.alloc_tile(tile_full)
        row_stat = pto.alloc_tile(tile_row_stat, valid_row=cHC, valid_col=c1)
        col_stat = pto.alloc_tile(tile_col_stat, valid_row=c1, valid_col=cHC)

        pre_sv = tile.subview(pre_full, [c0, c0], [1, HC])
        post_sv = tile.subview(post_full, [c0, c0], [1, HC])
        comb_kk = tile.subview(comb_full, [c0, c0], [HC, HC])
        scratch_kk = tile.subview(scratch_full, [c0, c0], [HC, HC])
        # 1xHC scratch slot reused as the reciprocal destination for sigmoid
        # (pto.trecip requires src != dst).
        recip_sv = tile.subview(scratch_full, [c0, c0], [1, HC])
        base_pre_sv = tile.subview(base_pre_full, [c0, c0], [1, HC])
        base_post_sv = tile.subview(base_post_full, [c0, c0], [1, HC])
        base_comb_sv = tile.subview(base_comb_full, [c0, c0], [HC, HC])

        # Load bases once per worker.
        base_pre_view = pto.slice_view(
            sv_row, source=tv_base_pre, offsets=[c0, c0], sizes=[c1, cHC]
        )
        base_post_view = pto.slice_view(
            sv_row, source=tv_base_post, offsets=[c0, c0], sizes=[c1, cHC]
        )
        base_comb_view = pto.slice_view(
            sv_kk, source=tv_base_comb, offsets=[c0, c0], sizes=[cHC, cHC]
        )
        pto.load(base_pre_view, base_pre_sv)
        pto.load(base_post_view, base_post_sv)
        pto.load(base_comb_view, base_comb_sv)

        # Load the three scalar scales.
        s0 = pto.load_scalar(s.float32, hc_scale_ptr, c0)
        s1 = pto.load_scalar(s.float32, hc_scale_ptr, c1)
        s2 = pto.load_scalar(s.float32, hc_scale_ptr, c2)

        for i in pto.range(wid, n, ncores):
            # ---- pre = sigmoid(mixes[i, :HC] * s0 + base[:HC]) + eps ----
            pre_in = pto.slice_view(
                sv_row, source=tv_mix_pre, offsets=[i, c0], sizes=[c1, cHC]
            )
            pto.load(pre_in, pre_sv)
            tile.muls(pre_sv, s0, pre_sv)
            tile.add(pre_sv, base_pre_sv, pre_sv)
            # sigmoid(x) = 1 / (1 + exp(-x))
            tile.muls(pre_sv, neg1_f, pre_sv)
            tile.exp(pre_sv, pre_sv)
            tile.adds(pre_sv, one_f, pre_sv)
            tile.reciprocal(pre_sv, recip_sv)
            tile.adds(recip_sv, eps, pre_sv)
            pre_out_view = pto.slice_view(
                sv_row, source=tv_pre_out, offsets=[i, c0], sizes=[c1, cHC]
            )
            pto.store(pre_sv, pre_out_view)

            # ---- post = 2 * sigmoid(mixes[i, HC:2HC] * s1 + base[HC:2HC]) ----
            post_in = pto.slice_view(
                sv_row, source=tv_mix_post, offsets=[i, c0], sizes=[c1, cHC]
            )
            pto.load(post_in, post_sv)
            tile.muls(post_sv, s1, post_sv)
            tile.add(post_sv, base_post_sv, post_sv)
            tile.muls(post_sv, neg1_f, post_sv)
            tile.exp(post_sv, post_sv)
            tile.adds(post_sv, one_f, post_sv)
            tile.reciprocal(post_sv, recip_sv)
            tile.muls(recip_sv, two_f, post_sv)
            post_out_view = pto.slice_view(
                sv_row, source=tv_post_out, offsets=[i, c0], sizes=[c1, cHC]
            )
            pto.store(post_sv, post_out_view)

            # ---- comb: scale + base, then sinkhorn-normalize ----
            comb_in_view = pto.slice_view(
                sv_kk,
                source=tv_mix_comb,
                offsets=[i, c0, c0],
                sizes=[c1, cHC, cHC],
            )
            pto.load(comb_in_view, comb_kk)
            tile.muls(comb_kk, s2, comb_kk)
            tile.add(comb_kk, base_comb_sv, comb_kk)

            # comb = softmax(comb, dim=-1) + eps
            tile.row_max(comb_kk, scratch_kk, row_stat)
            tile.row_expand_sub(comb_kk, row_stat, comb_kk)
            tile.exp(comb_kk, comb_kk)
            tile.row_sum(comb_kk, scratch_kk, row_stat)
            tile.row_expand_div(comb_kk, row_stat, comb_kk)
            tile.adds(comb_kk, eps, comb_kk)

            # comb /= (comb.sum(-2) + eps)
            tile.col_sum(comb_kk, scratch_kk, col_stat)
            tile.adds(col_stat, eps, col_stat)
            tile.col_expand_div(comb_kk, col_stat, comb_kk)

            for _ in pto.range(c1, const(SINKHORN_ITERS), c1):
                tile.row_sum(comb_kk, scratch_kk, row_stat)
                tile.adds(row_stat, eps, row_stat)
                tile.row_expand_div(comb_kk, row_stat, comb_kk)

                tile.col_sum(comb_kk, scratch_kk, col_stat)
                tile.adds(col_stat, eps, col_stat)
                tile.col_expand_div(comb_kk, col_stat, comb_kk)

            comb_out_view = pto.slice_view(
                sv_kk,
                source=tv_comb_out,
                offsets=[i, c0, c0],
                sizes=[c1, cHC, cHC],
            )
            pto.store(comb_kk, comb_out_view)


if __name__ == "__main__":
    print(hc_split_sinkhorn)
