"""
Sinkhorn normalization kernel — PTODSL builder (fp16 I/O, fp32 internal).

Algorithm (matches reference.cpp):
  For each (K, L) matrix in the batch of N:
    1. Initialise mu1[L] = mu2[K] = invMu1[L] = 1.0.
    2. For phase = 0..order:
         - Compute row & col standard deviations (unbiased) of cm/(mu1*mu2)
           in chunks of ROW_CHUNK rows.
         - phase == 0:  tgt = min(min(rStd), min(cStd)) + eps  [stored in tile]
         - phase >  0:  mu2 *= (rStd / tgt)^lr ;  mu1 *= (cStd / tgt)^lr ;
                        invMu1 = 1 / mu1
    3. Write matrix_out = cm / (mu1 * mu2) ; write mu1_out, mu2_out.

Design choices vs the hand-tuned reference:
  - No template specialisation on TileL: a single MAX_DIM=256 column stride
    is used for every L. Generated MLIR + ptoas auto-sync replaces the
    reference's manual flag/pipe management.
  - Native tile.log / tile.exp instead of the 2-term Pade approxLn.
  - col_expand_mul replaces the pre-tiled inv_mu1 buffer (one elementwise
    broadcast op instead of an explicit row-tile copy + flat TMUL).
  - Constraint: K must be a multiple of ROW_CHUNK = 8 (kernel returns
    early otherwise — same MAX_DIM upper-bound as reference).
"""

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

MAX_DIM = 256
ROW_CHUNK = 8


def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    i32 = pto.int32

    ptr_fp16 = pto.PtrType(fp16)

    tensor2_fp16 = pto.TensorType(rank=2, dtype=fp16)

    chunk_sub_fp16 = pto.SubTensorType(shape=[ROW_CHUNK, MAX_DIM], dtype=fp16)
    row_sub_fp16 = pto.SubTensorType(shape=[1, MAX_DIM], dtype=fp16)

    # ---- VEC tile types ----
    row_vec_cfg = pto.TileBufConfig()  # default RowMajor
    col_vec_cfg = pto.TileBufConfig(blayout="ColMajor")

    # Row vector [1, MAX_DIM] RowMajor fp32 — for L-indexed quantities
    # (mu1, invMu1, colSum, colSqsum, scratchL, zeroL).
    row_vec_fp32 = pto.TileBufType(
        shape=[1, MAX_DIM],
        valid_shape=[1, -1],
        dtype=fp32,
        memory_space="VEC",
        config=row_vec_cfg,
    )
    row_vec_fp16 = pto.TileBufType(
        shape=[1, MAX_DIM],
        valid_shape=[1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=row_vec_cfg,
    )

    # Per-chunk static col-major tile [ROW_CHUNK, 1] — used as TROWSUM dst
    # and TROWEXPANDDIV rhs scratch. Both shape and valid are static, so any
    # tile.reshape between this and its row-major sibling is fully static
    # and exercises the working codegen path.
    chunk_col_fp32_st = pto.TileBufType(
        shape=[ROW_CHUNK, 1],
        dtype=fp32,
        memory_space="VEC",
        config=col_vec_cfg,
    )
    chunk_row_fp32_st = pto.TileBufType(
        shape=[1, ROW_CHUNK],
        dtype=fp32,
        memory_space="VEC",
        config=row_vec_cfg,
    )

    # 2D chunk tiles [ROW_CHUNK, MAX_DIM]
    chunk_fp16 = pto.TileBufType(
        shape=[ROW_CHUNK, MAX_DIM],
        valid_shape=[ROW_CHUNK, -1],
        dtype=fp16,
        memory_space="VEC",
        config=row_vec_cfg,
    )
    chunk_fp32 = pto.TileBufType(
        shape=[ROW_CHUNK, MAX_DIM],
        valid_shape=[ROW_CHUNK, -1],
        dtype=fp32,
        memory_space="VEC",
        config=row_vec_cfg,
    )

    # Scalar [1, 1] ColMajor — broadcast-target for row_expand_div / col_expand_div.
    # Use dynamic valid_shape so the tile carries GetValidRow/Col, which are
    # required by TADDS / TMIN / TRESHAPE on the static 1x1 corner case.
    scalar_col_fp32 = pto.TileBufType(
        shape=[8, 1],
        valid_shape=[-1, -1],
        dtype=fp32,
        memory_space="VEC",
        config=col_vec_cfg,
    )
    # Scalar [1, 1] RowMajor alias used for elementwise ops (TMin, TAddS,...)
    # which require row-major layout.
    scalar_row_fp32 = pto.TileBufType(
        shape=[1, 8],
        valid_shape=[-1, -1],
        dtype=fp32,
        memory_space="VEC",
        config=row_vec_cfg,
    )

    return locals()


def build_sinkhorn(fn_name="sinkhorn_fp16"):
    @to_ir_module(meta_data=meta_data)
    def _kernel(
        matrix_in_ptr: "ptr_fp16",
        matrix_out_ptr: "ptr_fp16",
        mu1_out_ptr: "ptr_fp16",
        mu2_out_ptr: "ptr_fp16",
        N_i32: "i32",
        K_i32: "i32",
        L_i32: "i32",
        order_i32: "i32",
        lr: "fp32",
        eps: "fp32",
        invK: "fp32",
        invL: "fp32",
        invK1: "fp32",
        invL1: "fp32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cMAX_DIM = const(MAX_DIM)
        cROW_CHUNK = const(ROW_CHUNK)
        f0 = const(0.0, s.float32)
        f1 = const(1.0, s.float32)

        N = s.index_cast(N_i32)
        K = s.index_cast(K_i32)
        L = s.index_cast(L_i32)
        order = s.index_cast(order_i32)

        with pto.vector_section():
            # Bounds: 0 < K, L <= MAX_DIM ; K must be a multiple of ROW_CHUNK.
            ok = (
                (K > c0)
                & (L > c0)
                & (cMAX_DIM >= K)
                & (cMAX_DIM >= L)
                & s.eq(K % cROW_CHUNK, c0)
            )
            with pto.if_context(ok):
                cid = pto.get_block_idx()
                sub_bid = pto.get_subblock_idx()
                sub_bnum = pto.get_subblock_num()
                num_blocks = pto.get_block_num()
                wid = s.index_cast(cid * sub_bnum + sub_bid)
                num_workers = s.index_cast(num_blocks * sub_bnum)

                # ---- Allocate UB tiles (per worker, reused across batches) ----
                # K-indexed quantities are RowMajor [1, MAX_DIM] valid_col=K so
                # all subsequent elementwise ops (mul/sub/maxs/sqrt/log/exp/min)
                # work natively without going through a dynamic-valid reshape.
                mu1 = pto.alloc_tile(row_vec_fp32, valid_col=L)
                mu2 = pto.alloc_tile(row_vec_fp32, valid_col=K)
                invMu1 = pto.alloc_tile(row_vec_fp32, valid_col=L)

                colSum = pto.alloc_tile(row_vec_fp32, valid_col=L)
                colSqsum = pto.alloc_tile(row_vec_fp32, valid_col=L)
                rowSum = pto.alloc_tile(row_vec_fp32, valid_col=K)
                rowSqsum = pto.alloc_tile(row_vec_fp32, valid_col=K)

                scratchL = pto.alloc_tile(row_vec_fp32, valid_col=L)
                scratchK = pto.alloc_tile(row_vec_fp32, valid_col=K)

                chunkH = pto.alloc_tile(chunk_fp16, valid_col=L)
                chunkF = pto.alloc_tile(chunk_fp32, valid_col=L)
                chunkTmp = pto.alloc_tile(chunk_fp32, valid_col=L)

                # Per-chunk static col-major scratch (TROWSUM dst, TROWEXPANDDIV
                # rhs). [ROW_CHUNK, 1] both shape and valid fully static.
                rsumScratch = pto.alloc_tile(chunk_col_fp32_st)
                rsqScratch = pto.alloc_tile(chunk_col_fp32_st)

                # Static [1, ROW_CHUNK] row-major staging tile used to copy
                # the dynamic mu2[jg : jg + ROW_CHUNK] subview into a tile
                # whose storage Numel matches the [ROW_CHUNK, 1] col-major
                # sibling, so the subsequent tile.reshape passes the
                # codegen TRESHAPE byte-size static_assert.
                mu2RowStatic = pto.alloc_tile(chunk_row_fp32_st)

                tgtScalar = pto.alloc_tile(scalar_col_fp32, valid_row=c1, valid_col=c1)
                rMinTile = pto.alloc_tile(scalar_col_fp32, valid_row=c1, valid_col=c1)
                cMinTile = pto.alloc_tile(scalar_col_fp32, valid_row=c1, valid_col=c1)

                # Output staging tiles (fp16)
                mu1H = pto.alloc_tile(row_vec_fp16, valid_col=L)
                mu2H = pto.alloc_tile(row_vec_fp16, valid_col=K)

                # ---- Tensor views (rank-2 for matrix_in/out, rank-1 for mu*) ----
                NK = N * K
                tv_in = pto.as_tensor(
                    tensor2_fp16,
                    ptr=matrix_in_ptr,
                    shape=[NK, L],
                    strides=[L, c1],
                )
                tv_out = pto.as_tensor(
                    tensor2_fp16,
                    ptr=matrix_out_ptr,
                    shape=[NK, L],
                    strides=[L, c1],
                )
                tv_mu1 = pto.as_tensor(
                    tensor2_fp16,
                    ptr=mu1_out_ptr,
                    shape=[N, L],
                    strides=[L, c1],
                )
                tv_mu2 = pto.as_tensor(
                    tensor2_fp16,
                    ptr=mu2_out_ptr,
                    shape=[N, K],
                    strides=[K, c1],
                )

                # ============================================================
                #  Per-batch loop — workers split N across all vector cores.
                # ============================================================
                for bi in pto.range(wid, N, num_workers):
                    # Init mu1, mu2, invMu1 to all-ones via muls(.,0)+adds(.,1).
                    tile.muls(mu1, f0, mu1)
                    tile.adds(mu1, f1, mu1)
                    tile.muls(mu2, f0, mu2)
                    tile.adds(mu2, f1, mu2)
                    tile.muls(invMu1, f0, invMu1)
                    tile.adds(invMu1, f1, invMu1)

                    bi_row_off = bi * K  # row offset of this batch in tv_in/out

                    # ----------------------------------------------------------
                    #  Phase loop: phase 0 sets tgt; phases 1..order update mu.
                    # ----------------------------------------------------------
                    for phase in pto.range(c0, order + c1, c1):
                        # Reset col accumulators.
                        tile.muls(colSum, f0, colSum)
                        tile.muls(colSqsum, f0, colSqsum)

                        # Stream matrix in ROW_CHUNK-row chunks.
                        for jg in pto.range(c0, K, cROW_CHUNK):
                            # Load chunk fp16 [ROW_CHUNK, L] from GM.
                            chunk_view = pto.slice_view(
                                chunk_sub_fp16,
                                source=tv_in,
                                offsets=[bi_row_off + jg, c0],
                                sizes=[cROW_CHUNK, L],
                            )
                            pto.load(chunk_view, chunkH)

                            # fp16 -> fp32
                            tile.cvt(chunkH, chunkF)

                            # Build a col-major [ROW_CHUNK, 1] static view of
                            # mu2[jg : jg + ROW_CHUNK]. Subviewing mu2 keeps
                            # the parent's storage Numel (=MAX_DIM), so a
                            # direct reshape to [ROW_CHUNK, 1] would fail the
                            # codegen TRESHAPE byte-size static_assert. Copy
                            # the 8 elements into a static [1, ROW_CHUNK]
                            # tile (storage Numel=8) first via tile.muls
                            # with multiplier 1.0, then reshape that to the
                            # col-major sibling (storage Numel matches).
                            mu2_row_chunk = pto.subview(
                                mu2, offsets=[c0, jg], sizes=[1, ROW_CHUNK]
                            )
                            tile.muls(mu2_row_chunk, f1, mu2RowStatic)
                            mu2_col_chunk = tile.reshape(
                                chunk_col_fp32_st, mu2RowStatic
                            )
                            tile.row_expand_div(chunkF, mu2_col_chunk, chunkF)

                            # Multiply each col by invMu1[c] (broadcast row-vec).
                            tile.col_expand_mul(chunkF, invMu1, chunkF)

                            # Row-sum into per-chunk col scratch, then scatter
                            # into row-major rowSum[jg : jg + ROW_CHUNK].
                            tile.row_sum(chunkF, chunkTmp, rsumScratch)
                            rsum_row_view = tile.reshape(chunk_row_fp32_st, rsumScratch)
                            rowSum_chunk = pto.subview(
                                rowSum, offsets=[c0, jg], sizes=[1, ROW_CHUNK]
                            )
                            tile.muls(rsum_row_view, f1, rowSum_chunk)

                            # Col-sum: accumulate across chunks.
                            tile.col_sum(chunkF, chunkTmp, scratchL, is_binary=True)
                            tile.add(colSum, scratchL, colSum)

                            # Square chunk for sq-sum stats.
                            tile.mul(chunkF, chunkF, chunkF)

                            tile.row_sum(chunkF, chunkTmp, rsqScratch)
                            rsq_row_view = tile.reshape(chunk_row_fp32_st, rsqScratch)
                            rowSq_chunk = pto.subview(
                                rowSqsum, offsets=[c0, jg], sizes=[1, ROW_CHUNK]
                            )
                            tile.muls(rsq_row_view, f1, rowSq_chunk)

                            tile.col_sum(chunkF, chunkTmp, scratchL, is_binary=True)
                            tile.add(colSqsum, scratchL, colSqsum)

                        # ---- Finalise row std (unbiased): rStd = sqrt(max(0,
                        #         (rSqsum - rSum^2 * invL) * invL1)) ----
                        tile.mul(rowSum, rowSum, scratchK)
                        tile.muls(scratchK, invL, scratchK)
                        tile.sub(rowSqsum, scratchK, rowSqsum)
                        tile.muls(rowSqsum, invL1, rowSqsum)
                        tile.maxs(rowSqsum, f0, rowSqsum)
                        tile.sqrt(rowSqsum, rowSqsum)

                        # ---- Finalise col std (unbiased) ----
                        tile.mul(colSum, colSum, scratchL)
                        tile.muls(scratchL, invK, scratchL)
                        tile.sub(colSqsum, scratchL, colSqsum)
                        tile.muls(colSqsum, invK1, colSqsum)
                        tile.maxs(colSqsum, f0, colSqsum)
                        tile.sqrt(colSqsum, colSqsum)

                        with pto.if_context(s.eq(phase, c0), has_else=True) as br:
                            # ---- Phase 0: tgt = min(rStd_min, cStd_min) + eps ----
                            tile.row_min(rowSqsum, scratchK, rMinTile)
                            tile.row_min(colSqsum, scratchL, cMinTile)
                            # TMin / TAddS need row-major: alias the col scalars
                            # via tile.reshape (static [1,1] -> [1,1] is a no-op
                            # codegen-wise but the source has dynamic valid 1x1
                            # set explicitly at alloc).
                            rMin_r = tile.reshape(scalar_row_fp32, rMinTile)
                            cMin_r = tile.reshape(scalar_row_fp32, cMinTile)
                            tgt_r = tile.reshape(scalar_row_fp32, tgtScalar)
                            tile.min(rMin_r, cMin_r, tgt_r)
                            tile.adds(tgt_r, eps, tgt_r)
                        with br.else_context():
                            # ---- Phase >0: mu2 *= (rStd/tgt)^lr ----
                            # rowSqsum is RowMajor [1, K], tgtScalar is ColMajor
                            # [8, 1] valid=1x1 — TROWEXPANDDIV broadcasts the
                            # 1-element col-vec across the K columns of the
                            # 1-row dst (= scalar division).
                            tile.row_expand_div(rowSqsum, tgtScalar, rowSqsum)
                            tile.maxs(rowSqsum, const(1e-12, s.float32), rowSqsum)
                            tile.log(rowSqsum, rowSqsum)
                            tile.muls(rowSqsum, lr, rowSqsum)
                            tile.exp(rowSqsum, rowSqsum)
                            tile.mul(mu2, rowSqsum, mu2)

                            # ---- mu1 *= (cStd/tgt)^lr ----
                            tile.row_expand_div(colSqsum, tgtScalar, colSqsum)
                            tile.maxs(colSqsum, const(1e-12, s.float32), colSqsum)
                            tile.log(colSqsum, colSqsum)
                            tile.muls(colSqsum, lr, colSqsum)
                            tile.exp(colSqsum, colSqsum)
                            tile.mul(mu1, colSqsum, mu1)

                            # invMu1 = 1 / mu1
                            tile.reciprocal(mu1, invMu1)

                    # ============================================================
                    #  Write matrix_out = cm / (mu1 * mu2)
                    # ============================================================
                    for jg in pto.range(c0, K, cROW_CHUNK):
                        chunk_view = pto.slice_view(
                            chunk_sub_fp16,
                            source=tv_in,
                            offsets=[bi_row_off + jg, c0],
                            sizes=[cROW_CHUNK, L],
                        )
                        pto.load(chunk_view, chunkH)
                        tile.cvt(chunkH, chunkF)

                        mu2_row_chunk = pto.subview(
                            mu2, offsets=[c0, jg], sizes=[1, ROW_CHUNK]
                        )
                        tile.muls(mu2_row_chunk, f1, mu2RowStatic)
                        mu2_col_chunk = tile.reshape(chunk_col_fp32_st, mu2RowStatic)
                        tile.row_expand_div(chunkF, mu2_col_chunk, chunkF)
                        tile.col_expand_mul(chunkF, invMu1, chunkF)

                        tile.cvt(chunkF, chunkH, rmode="cast_rint")

                        out_view = pto.slice_view(
                            chunk_sub_fp16,
                            source=tv_out,
                            offsets=[bi_row_off + jg, c0],
                            sizes=[cROW_CHUNK, L],
                        )
                        pto.store(chunkH, out_view)

                    # ---- Write mu1_out (length L per batch) ----
                    tile.cvt(mu1, mu1H, rmode="cast_rint")
                    mu1_view = pto.slice_view(
                        row_sub_fp16,
                        source=tv_mu1,
                        offsets=[bi, c0],
                        sizes=[c1, L],
                    )
                    pto.store(mu1H, mu1_view)

                    # ---- Write mu2_out (length K per batch) ----
                    # mu2 is now RowMajor [1, MAX_DIM] valid_col=K — direct cvt.
                    tile.cvt(mu2, mu2H, rmode="cast_rint")
                    mu2_view = pto.slice_view(
                        row_sub_fp16,
                        source=tv_mu2,
                        offsets=[bi, c0],
                        sizes=[c1, K],
                    )
                    pto.store(mu2H, mu2_view)

    _ = fn_name
    return _kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fn-name", default="sinkhorn_fp16")
    args = parser.parse_args()
    print(build_sinkhorn(fn_name=args.fn_name))
