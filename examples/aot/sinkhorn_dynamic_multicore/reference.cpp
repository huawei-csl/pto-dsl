/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/

/**
 * Sinkhorn normalization kernel for Ascend NPU (fp16 I/O, fp32 internal).
 *
 * Algorithm:
 *   For each (K, L) matrix in the batch:
 *   1. Compute row/col standard deviations of cm / (mu1 * mu2).
 *   2. Target = min(all stds) + eps.
 *   3. Iterate: mu *= pow(std / tgt, lr) for each row/col.
 *   4. Output: matrix_out = cm / (mu1 * mu2).
 *
 * Performance design:
 *   - Templated on TileL (column width) so the 2D row stride matches the
 *     data.  flat TCVT / TMUL operate on cr*TileL elements instead of
 *     cr*MAX_DIM — up to 8x less work for small L.
 *   - inv_mu1 pre-tiled into a 2D flat buffer once per phase; a single
 *     flat TMUL per chunk replaces 8 row-by-row TDIVs.
 *   - pow(x,lr) via 2-term Pade approxLn + TEXP (8 barriers, not 14).
 */

#include <pto/pto-inst.hpp>

// clang-format off
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
// clang-format on

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))
#define ALIGN_UP(x, y) (DIV_ROUNDUP((x), (y)) * (y))

constexpr uint32_t UB_USABLE_BYTES = 192 * 1024;
constexpr uint32_t MAX_DIM = 256;
constexpr uint32_t ROW_CHUNK = 8;
constexpr uint32_t TILE_ALIGN = 16;

// ---------- UB layout (sized for worst case MAX_DIM) ----------
namespace UbOfs {
constexpr unsigned MU1 = 0x00000;
constexpr unsigned MU2 = MU1 + MAX_DIM * sizeof(float);
constexpr unsigned INV_MU1 = MU2 + MAX_DIM * sizeof(float);
constexpr unsigned ROW_SUM = INV_MU1 + MAX_DIM * sizeof(float);
constexpr unsigned ROW_SQSUM = ROW_SUM + MAX_DIM * sizeof(float);
constexpr unsigned COL_SUM = ROW_SQSUM + MAX_DIM * sizeof(float);
constexpr unsigned COL_SQSUM = COL_SUM + MAX_DIM * sizeof(float);
constexpr unsigned CHUNK_HALF = COL_SQSUM + MAX_DIM * sizeof(float);
constexpr unsigned CHUNK_FP32 = CHUNK_HALF + ROW_CHUNK * MAX_DIM * sizeof(half);
constexpr unsigned CHUNK_TMP = CHUNK_FP32 + ROW_CHUNK * MAX_DIM * sizeof(float);
constexpr unsigned SCRATCH = CHUNK_TMP + ROW_CHUNK * MAX_DIM * sizeof(float);
constexpr unsigned SCALAR_A = SCRATCH + MAX_DIM * sizeof(float);
constexpr unsigned SCALAR_B = SCALAR_A + 32;
constexpr unsigned ZERO_VEC = SCALAR_B + 32;
constexpr unsigned LN_TMP1 = ZERO_VEC + MAX_DIM * sizeof(float);
constexpr unsigned LN_TMP2 = LN_TMP1 + MAX_DIM * sizeof(float);
constexpr unsigned INV_MU1_TILED = LN_TMP2 + MAX_DIM * sizeof(float);
constexpr unsigned TOTAL = INV_MU1_TILED + ROW_CHUNK * MAX_DIM * sizeof(float);
}  // namespace UbOfs

static_assert(UbOfs::TOTAL <= UB_USABLE_BYTES,
              "Sinkhorn UB layout exceeds 192 KB.");

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

// ---------- Tile type aliases ----------
using StrideDim5 = pto::Stride<1, 1, 1, 1, 1>;

template <typename T, uint32_t N>
using Vec1D = Tile<TileType::Vec, T, 1, N, BLayout::RowMajor, -1, -1>;

template <typename T, uint32_t N>
using Global1D = GlobalTensor<T, Shape<1, 1, 1, 1, N>, StrideDim5>;

using DynStride = Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T>
using Shape2D = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
template <typename T, uint32_t R, uint32_t C>
using Tile2D =
    Tile<TileType::Vec, T, R, C, BLayout::RowMajor, DYNAMIC, DYNAMIC>;
template <typename T, uint32_t C>
using Global2D = GlobalTensor<T, Shape2D<T>, DynStride, Layout::ND>;

using ScalarCol = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, -1, -1>;

template <typename T, uint32_t R>
using ColVec =
    Tile<TileType::Vec, T, R, 1, BLayout::ColMajor, DYNAMIC, DYNAMIC>;

// ---------- Pipe helpers ----------
AICORE inline void initPipeFlags() {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

AICORE inline void drainPipeFlags() {
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

// ---------- 2-term Pade approxLn(x) for x > 0 ----------
template <uint32_t MaxN>
AICORE void approxLn(uint32_t N, unsigned dataOfs) {
  Vec1D<float, MaxN> v(1, N);
  Vec1D<float, MaxN> t1(1, N);
  Vec1D<float, MaxN> r(1, N);
  TASSIGN(v, dataOfs);
  TASSIGN(t1, UbOfs::LN_TMP1);
  TASSIGN(r, UbOfs::SCRATCH);

  TADDS(t1, v, -1.0f);
  pipe_barrier(PIPE_V);
  TADDS(v, v, 1.0f);
  pipe_barrier(PIPE_V);
  TDIV(r, t1, v);
  pipe_barrier(PIPE_V);
  TMUL(t1, r, r);
  pipe_barrier(PIPE_V);
  TMULS(v, t1, 1.0f / 3.0f);
  pipe_barrier(PIPE_V);
  TADDS(v, v, 1.0f);
  pipe_barrier(PIPE_V);
  TMUL(v, v, r);
  pipe_barrier(PIPE_V);
  TMULS(v, v, 2.0f);
  pipe_barrier(PIPE_V);
}

// ---------- Tile inv_mu1 into 2D flat buffer ----------
template <uint32_t TileL>
AICORE void tileInvMu1(uint32_t La) {
  constexpr unsigned rowBytes = TileL * sizeof(float);
  Vec1D<float, MAX_DIM> src(1, La);
  TASSIGN(src, UbOfs::INV_MU1);
  for (uint32_t r = 0; r < ROW_CHUNK; ++r) {
    Vec1D<float, TileL> dst(1, La);
    TASSIGN(dst, UbOfs::INV_MU1_TILED + r * rowBytes);
    TMULS(dst, src, 1.0f);
    pipe_barrier(PIPE_V);
  }
}

// ---------- Main kernel, templated on tile column width ----------
template <typename T, uint32_t TileL>
AICORE void runSinkhornImpl(__gm__ T *matrix_in, __gm__ T *matrix_out,
                            __gm__ T *mu1_out, __gm__ T *mu2_out, uint32_t N,
                            uint32_t K, uint32_t L, uint32_t La, uint32_t Ka,
                            uint32_t order, float lr, float eps, float invK,
                            float invL, float invK1, float invL1) {
  const uint32_t num_workers = get_block_num() * get_subblockdim();
  const uint32_t wid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t KL = K * L;

  initPipeFlags();

  for (uint32_t bi = wid; bi < N; bi += num_workers) {
    __gm__ T *cm = matrix_in + static_cast<size_t>(bi) * KL;

    // ---- init ----
    Vec1D<float, MAX_DIM> mu1(1, La);
    Vec1D<float, MAX_DIM> mu2(1, Ka);
    Vec1D<float, MAX_DIM> invMu1(1, La);
    TASSIGN(mu1, UbOfs::MU1);
    TASSIGN(mu2, UbOfs::MU2);
    TASSIGN(invMu1, UbOfs::INV_MU1);
    TEXPANDS(mu1, 1.0f);
    pipe_barrier(PIPE_V);
    TEXPANDS(mu2, 1.0f);
    pipe_barrier(PIPE_V);
    TEXPANDS(invMu1, 1.0f);
    pipe_barrier(PIPE_V);

    {
      uint32_t zLen = Ka > La ? Ka : La;
      Vec1D<float, MAX_DIM> zeroVec(1, zLen);
      TASSIGN(zeroVec, UbOfs::ZERO_VEC);
      TEXPANDS(zeroVec, 0.0f);
      pipe_barrier(PIPE_V);
    }

    tileInvMu1<TileL>(La);

    // ============================================================
    //  Phase loop
    // ============================================================
    for (uint32_t phase = 0; phase <= order; ++phase) {
      Vec1D<float, MAX_DIM> colSum(1, La);
      Vec1D<float, MAX_DIM> colSqsum(1, La);
      TASSIGN(colSum, UbOfs::COL_SUM);
      TASSIGN(colSqsum, UbOfs::COL_SQSUM);
      TEXPANDS(colSum, 0.0f);
      pipe_barrier(PIPE_V);
      TEXPANDS(colSqsum, 0.0f);
      pipe_barrier(PIPE_V);

      // ---- stream matrix in ROW_CHUNK-row chunks ----
      for (uint32_t jg = 0; jg < K; jg += ROW_CHUNK) {
        const uint32_t cr = (jg + ROW_CHUNK <= K) ? ROW_CHUNK : (K - jg);
        const uint32_t flat = cr * TileL;  // tight: stride = TileL

        // Load
        Tile2D<T, ROW_CHUNK, TileL> chunkHalf(cr, La);
        TASSIGN(chunkHalf, UbOfs::CHUNK_HALF);
        Shape2D<T> chunkShape(cr, L);
        DynStride chunkStride(L);
        Global2D<T, TileL> chunkGlobal(cm + jg * L, chunkShape, chunkStride);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(chunkHalf, chunkGlobal);
        pipe_barrier(PIPE_ALL);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // fp16 -> fp32
        Vec1D<T, ROW_CHUNK * TileL> halfFlat(1, flat);
        Vec1D<float, ROW_CHUNK * TileL> fp32Flat(1, flat);
        TASSIGN(halfFlat, UbOfs::CHUNK_HALF);
        TASSIGN(fp32Flat, UbOfs::CHUNK_FP32);
        TCVT(fp32Flat, halfFlat, RoundMode::CAST_NONE);
        pipe_barrier(PIPE_V);

        Tile2D<float, ROW_CHUNK, TileL> chunk(cr, La);
        TASSIGN(chunk, UbOfs::CHUNK_FP32);

        // Divide by mu2
        ColVec<float, ROW_CHUNK> mu2Sub(cr, 1);
        TASSIGN(mu2Sub, UbOfs::MU2 + jg * sizeof(float));
        TROWEXPANDDIV(chunk, chunk, mu2Sub);
        pipe_barrier(PIPE_V);

        // Multiply by tiled inv_mu1 (1 flat TMUL)
        {
          Vec1D<float, ROW_CHUNK * TileL> cFlat(1, flat);
          Vec1D<float, ROW_CHUNK * TileL> iFlat(1, flat);
          TASSIGN(cFlat, UbOfs::CHUNK_FP32);
          TASSIGN(iFlat, UbOfs::INV_MU1_TILED);
          TMUL(cFlat, cFlat, iFlat);
          pipe_barrier(PIPE_V);
        }

        // Row stats
        Tile2D<float, ROW_CHUNK, TileL> tmp(cr, La);
        TASSIGN(tmp, UbOfs::CHUNK_TMP);

        ColVec<float, ROW_CHUNK> rowSumPart(cr, 1);
        TASSIGN(rowSumPart, UbOfs::ROW_SUM + jg * sizeof(float));
        TROWSUM(rowSumPart, chunk, tmp);
        pipe_barrier(PIPE_V);

        Vec1D<float, MAX_DIM> partCol(1, La);
        TASSIGN(partCol, UbOfs::SCRATCH);
        TCOLSUM(partCol, chunk, tmp, false);
        pipe_barrier(PIPE_V);
        if (jg == 0) {
          TMULS(colSum, partCol, 1.0f);
        } else {
          TADD(colSum, colSum, partCol);
        }
        pipe_barrier(PIPE_V);

        TMUL(chunk, chunk, chunk);
        pipe_barrier(PIPE_V);

        ColVec<float, ROW_CHUNK> rowSqPart(cr, 1);
        TASSIGN(rowSqPart, UbOfs::ROW_SQSUM + jg * sizeof(float));
        TROWSUM(rowSqPart, chunk, tmp);
        pipe_barrier(PIPE_V);

        TCOLSUM(partCol, chunk, tmp, false);
        pipe_barrier(PIPE_V);
        if (jg == 0) {
          TMULS(colSqsum, partCol, 1.0f);
        } else {
          TADD(colSqsum, colSqsum, partCol);
        }
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      }  // chunk loop

      // ---- Finalise row_std ----
      Vec1D<float, MAX_DIM> rSum(1, Ka);
      Vec1D<float, MAX_DIM> rStd(1, Ka);
      Vec1D<float, MAX_DIM> scrK(1, Ka);
      Vec1D<float, MAX_DIM> zeroK(1, Ka);
      TASSIGN(rSum, UbOfs::ROW_SUM);
      TASSIGN(rStd, UbOfs::ROW_SQSUM);
      TASSIGN(scrK, UbOfs::SCRATCH);
      TASSIGN(zeroK, UbOfs::ZERO_VEC);

      TMUL(scrK, rSum, rSum);
      pipe_barrier(PIPE_V);
      TMULS(scrK, scrK, invL);
      pipe_barrier(PIPE_V);
      TSUB(rStd, rStd, scrK);
      pipe_barrier(PIPE_V);
      TMULS(rStd, rStd, invL1);
      pipe_barrier(PIPE_V);
      TMAX(rStd, rStd, zeroK);
      pipe_barrier(PIPE_V);
      TSQRT(rStd, rStd);
      pipe_barrier(PIPE_V);

      // ---- Finalise col_std ----
      Vec1D<float, MAX_DIM> cSum(1, La);
      Vec1D<float, MAX_DIM> cStd(1, La);
      Vec1D<float, MAX_DIM> scrL(1, La);
      Vec1D<float, MAX_DIM> zeroL(1, La);
      TASSIGN(cSum, UbOfs::COL_SUM);
      TASSIGN(cStd, UbOfs::COL_SQSUM);
      TASSIGN(scrL, UbOfs::SCRATCH);
      TASSIGN(zeroL, UbOfs::ZERO_VEC);

      TMUL(scrL, cSum, cSum);
      pipe_barrier(PIPE_V);
      TMULS(scrL, scrL, invK);
      pipe_barrier(PIPE_V);
      TSUB(cStd, cStd, scrL);
      pipe_barrier(PIPE_V);
      TMULS(cStd, cStd, invK1);
      pipe_barrier(PIPE_V);
      TMAX(cStd, cStd, zeroL);
      pipe_barrier(PIPE_V);
      TSQRT(cStd, cStd);
      pipe_barrier(PIPE_V);

      if (phase == 0) {
        Vec1D<float, MAX_DIM> rStd1D(1, Ka);
        Vec1D<float, MAX_DIM> rMinTmp(1, Ka);
        ScalarCol rMinS(1, 1);
        TASSIGN(rStd1D, UbOfs::ROW_SQSUM);
        TASSIGN(rMinTmp, UbOfs::SCRATCH);
        TASSIGN(rMinS, UbOfs::SCALAR_A);
        TROWMIN(rMinS, rStd1D, rMinTmp);
        pipe_barrier(PIPE_V);

        Vec1D<float, MAX_DIM> cStd1D(1, La);
        Vec1D<float, MAX_DIM> cMinTmp(1, La);
        ScalarCol cMinS(1, 1);
        TASSIGN(cStd1D, UbOfs::COL_SQSUM);
        TASSIGN(cMinTmp, UbOfs::SCRATCH);
        TASSIGN(cMinS, UbOfs::SCALAR_B);
        TROWMIN(cMinS, cStd1D, cMinTmp);
        pipe_barrier(PIPE_V);

        Vec1D<float, 8> sA(1, 1);
        Vec1D<float, 8> sB(1, 1);
        TASSIGN(sA, UbOfs::SCALAR_A);
        TASSIGN(sB, UbOfs::SCALAR_B);
        TMIN(sA, sA, sB);
        pipe_barrier(PIPE_V);
        TADDS(sA, sA, eps);
        pipe_barrier(PIPE_V);
      } else {
        // ---- mu update ----
        ScalarCol tgtCol(1, 1);
        TASSIGN(tgtCol, UbOfs::SCALAR_A);

        // mu2 *= pow(row_std / tgt, lr)
        Vec1D<float, MAX_DIM> rStdUpd(1, Ka);
        TASSIGN(rStdUpd, UbOfs::ROW_SQSUM);
        TROWEXPANDDIV(rStdUpd, rStdUpd, tgtCol);
        pipe_barrier(PIPE_V);
        {
          Vec1D<float, MAX_DIM> epsVec(1, Ka);
          TASSIGN(epsVec, UbOfs::LN_TMP1);
          TEXPANDS(epsVec, 1e-12f);
          pipe_barrier(PIPE_V);
          TMAX(rStdUpd, rStdUpd, epsVec);
          pipe_barrier(PIPE_V);
        }
        approxLn<MAX_DIM>(Ka, UbOfs::ROW_SQSUM);
        TASSIGN(rStdUpd, UbOfs::ROW_SQSUM);
        TMULS(rStdUpd, rStdUpd, lr);
        pipe_barrier(PIPE_V);
        TEXP(rStdUpd, rStdUpd);
        pipe_barrier(PIPE_V);
        Vec1D<float, MAX_DIM> mu2Upd(1, Ka);
        TASSIGN(mu2Upd, UbOfs::MU2);
        TMUL(mu2Upd, mu2Upd, rStdUpd);
        pipe_barrier(PIPE_V);

        // mu1 *= pow(col_std / tgt, lr)
        Vec1D<float, MAX_DIM> cStdUpd(1, La);
        TASSIGN(cStdUpd, UbOfs::COL_SQSUM);
        TROWEXPANDDIV(cStdUpd, cStdUpd, tgtCol);
        pipe_barrier(PIPE_V);
        {
          Vec1D<float, MAX_DIM> epsVec(1, La);
          TASSIGN(epsVec, UbOfs::LN_TMP1);
          TEXPANDS(epsVec, 1e-12f);
          pipe_barrier(PIPE_V);
          TMAX(cStdUpd, cStdUpd, epsVec);
          pipe_barrier(PIPE_V);
        }
        approxLn<MAX_DIM>(La, UbOfs::COL_SQSUM);
        TASSIGN(cStdUpd, UbOfs::COL_SQSUM);
        TMULS(cStdUpd, cStdUpd, lr);
        pipe_barrier(PIPE_V);
        TEXP(cStdUpd, cStdUpd);
        pipe_barrier(PIPE_V);
        Vec1D<float, MAX_DIM> mu1Upd(1, La);
        TASSIGN(mu1Upd, UbOfs::MU1);
        TMUL(mu1Upd, mu1Upd, cStdUpd);
        pipe_barrier(PIPE_V);

        // Refresh inv_mu1 and re-tile
        Vec1D<float, MAX_DIM> ones(1, La);
        TASSIGN(ones, UbOfs::LN_TMP1);
        TEXPANDS(ones, 1.0f);
        pipe_barrier(PIPE_V);
        Vec1D<float, MAX_DIM> newInv(1, La);
        TASSIGN(newInv, UbOfs::INV_MU1);
        TASSIGN(mu1Upd, UbOfs::MU1);
        TDIV(newInv, ones, mu1Upd);
        pipe_barrier(PIPE_V);
        tileInvMu1<TileL>(La);
      }
    }  // phase loop

    // ============================================================
    //  Write output
    // ============================================================
    __gm__ T *out = matrix_out + static_cast<size_t>(bi) * KL;

    for (uint32_t jg = 0; jg < K; jg += ROW_CHUNK) {
      const uint32_t cr = (jg + ROW_CHUNK <= K) ? ROW_CHUNK : (K - jg);
      const uint32_t flat = cr * TileL;

      Tile2D<T, ROW_CHUNK, TileL> chunkHalf(cr, La);
      TASSIGN(chunkHalf, UbOfs::CHUNK_HALF);
      Shape2D<T> inShape(cr, L);
      DynStride inStride(L);
      Global2D<T, TileL> inGlobal(cm + jg * L, inShape, inStride);

      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(chunkHalf, inGlobal);
      pipe_barrier(PIPE_ALL);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      Vec1D<T, ROW_CHUNK * TileL> hFlat(1, flat);
      Vec1D<float, ROW_CHUNK * TileL> fFlat(1, flat);
      TASSIGN(hFlat, UbOfs::CHUNK_HALF);
      TASSIGN(fFlat, UbOfs::CHUNK_FP32);
      TCVT(fFlat, hFlat, RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);

      Tile2D<float, ROW_CHUNK, TileL> chunk(cr, La);
      TASSIGN(chunk, UbOfs::CHUNK_FP32);

      ColVec<float, ROW_CHUNK> mu2Sub(cr, 1);
      TASSIGN(mu2Sub, UbOfs::MU2 + jg * sizeof(float));
      TROWEXPANDDIV(chunk, chunk, mu2Sub);
      pipe_barrier(PIPE_V);

      {
        Vec1D<float, ROW_CHUNK * TileL> cFlat(1, flat);
        Vec1D<float, ROW_CHUNK * TileL> iFlat(1, flat);
        TASSIGN(cFlat, UbOfs::CHUNK_FP32);
        TASSIGN(iFlat, UbOfs::INV_MU1_TILED);
        TMUL(cFlat, cFlat, iFlat);
        pipe_barrier(PIPE_V);
      }

      TCVT(hFlat, fFlat, RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);

      Shape2D<T> outShape(cr, L);
      DynStride outStride(L);
      Global2D<T, TileL> outGlobal(out + jg * L, outShape, outStride);

      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(outGlobal, chunkHalf);
      pipe_barrier(PIPE_ALL);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }

    // ---- Write mu1_out ----
    {
      Vec1D<float, MAX_DIM> mu1F(1, La);
      Vec1D<T, MAX_DIM> mu1H(1, La);
      TASSIGN(mu1F, UbOfs::MU1);
      TASSIGN(mu1H, UbOfs::CHUNK_HALF);
      TCVT(mu1H, mu1F, RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);

      Global1D<T, MAX_DIM> mu1G(mu1_out + static_cast<size_t>(bi) * L);
      TASSIGN(mu1G, (mu1_out + static_cast<size_t>(bi) * L));

      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(mu1G, mu1H);
      pipe_barrier(PIPE_ALL);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    // ---- Write mu2_out ----
    {
      Vec1D<float, MAX_DIM> mu2F(1, Ka);
      Vec1D<T, MAX_DIM> mu2H(1, Ka);
      TASSIGN(mu2F, UbOfs::MU2);
      TASSIGN(mu2H, UbOfs::CHUNK_HALF);
      TCVT(mu2H, mu2F, RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);

      Global1D<T, MAX_DIM> mu2G(mu2_out + static_cast<size_t>(bi) * K);
      TASSIGN(mu2G, (mu2_out + static_cast<size_t>(bi) * K));

      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TSTORE(mu2G, mu2H);
      pipe_barrier(PIPE_ALL);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
  }  // bi loop

  drainPipeFlags();
}

// ---------- Dispatch to TileL-specialised impl ----------
template <typename T>
AICORE void runSinkhorn(__gm__ T *matrix_in, __gm__ T *matrix_out,
                        __gm__ T *mu1_out, __gm__ T *mu2_out, uint32_t N,
                        uint32_t K, uint32_t L, uint32_t order, float lr,
                        float eps, float invK, float invL, float invK1,
                        float invL1) {
  set_mask_norm();
  set_vector_mask(-1, -1);
  if (K == 0 || L == 0 || K > MAX_DIM || L > MAX_DIM) return;

  const uint32_t La = ALIGN_UP(L, TILE_ALIGN);
  const uint32_t Ka = ALIGN_UP(K, TILE_ALIGN);

  // Dispatch to tight-stride specialisation.
  // For La <= 32, the flat vectors are too short — barrier overhead dominates,
  // so wider stride (MAX_DIM) amortises better.  Specialise from La >= 64.
  switch (La) {
    case 64:
      runSinkhornImpl<T, 64>(matrix_in, matrix_out, mu1_out, mu2_out, N, K, L,
                             La, Ka, order, lr, eps, invK, invL, invK1, invL1);
      break;
    case 128:
      runSinkhornImpl<T, 128>(matrix_in, matrix_out, mu1_out, mu2_out, N, K, L,
                              La, Ka, order, lr, eps, invK, invL, invK1, invL1);
      break;
    default:
      // La <= 32 or La >= 192: use MAX_DIM stride (long flat vectors)
      runSinkhornImpl<T, MAX_DIM>(matrix_in, matrix_out, mu1_out, mu2_out, N, K,
                                  L, La, Ka, order, lr, eps, invK, invL, invK1,
                                  invL1);
      break;
  }
}

#endif  // __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

// ---------- Entry points ----------

extern "C" __global__ AICORE void sinkhorn_fp16(
    GM_ADDR matrix_in, GM_ADDR matrix_out, GM_ADDR mu1_out, GM_ADDR mu2_out,
    uint32_t N, uint32_t K, uint32_t L, uint32_t order, float lr, float eps,
    float invK, float invL, float invK1, float invL1) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  runSinkhorn<half>((__gm__ half *)matrix_in, (__gm__ half *)matrix_out,
                    (__gm__ half *)mu1_out, (__gm__ half *)mu2_out, N, K, L,
                    order, lr, eps, invK, invL, invK1, invL1);
#else
  (void)matrix_in;
  (void)matrix_out;
  (void)mu1_out;
  (void)mu2_out;
  (void)N;
  (void)K;
  (void)L;
  (void)order;
  (void)lr;
  (void)eps;
  (void)invK;
  (void)invL;
  (void)invK1;
  (void)invL1;
#endif
}

extern "C" void call_sinkhorn_kernel(uint32_t blockDim, void *stream,
                                     uint8_t *matrix_in, uint8_t *matrix_out,
                                     uint8_t *mu1_out, uint8_t *mu2_out,
                                     uint32_t N, uint32_t K, uint32_t L,
                                     uint32_t order, float lr, float eps,
                                     float invK, float invL, float invK1,
                                     float invL1) {
  sinkhorn_fp16<<<blockDim * 2, nullptr, stream>>>(
      matrix_in, matrix_out, mu1_out, mu2_out, N, K, L, order, lr, eps, invK,
      invL, invK1, invL1);
}
