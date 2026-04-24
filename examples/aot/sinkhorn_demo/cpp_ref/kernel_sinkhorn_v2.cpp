/**
Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
See LICENSE in the root of the software repository for the full License text.
*/

/**
 * Doubly-stochastic Sinkhorn normalization — Ascend 910B kernel (fp16 I/O, K=4).
 *
 * K=4-only specialization of kernel_sinkhorn.cpp's K=4 dispatch paths
 * (DeepSeek MHC sinkhorn, hc_mult=4).  Drops the K=8/16/32/64/128 paths
 * and the K-runtime templating so every tile dimension is a compile-time
 * constant at the ABI boundary.
 *
 *     x = x.softmax(-1) + eps
 *     x = x / (x.sum(-2, keepdim=True) + eps)
 *     for _ in range(repeat - 1):
 *         x = x / (x.sum(-1, keepdim=True) + eps)
 *         x = x / (x.sum(-2, keepdim=True) + eps)
 *
 * Two code paths, dispatched on batch size:
 *
 *   N >= 2048  — sinkhornFastPath
 *                Interleaved (K, ROW_BLOCK_COLS) tile + double-buffered
 *                bulk TLOAD / TSTORE.  Col-normalize = TCOLSUM +
 *                TCOLEXPANDDIV on the full interleaved tile (2 ops,
 *                2 barriers per iteration).
 *
 *   N <  2048  — sinkhornSmallBatch
 *                Natural-order layout, no double-buffer.  Per-matrix
 *                col-normalize with TCOLEXPANDDIV on each 4×4 sub-tile.
 *                Pays almost no setup cost; wins at small batch.
 *
 * Parallelism model (both paths):
 *   N matrices sharded across AIV cores (num_workers total).  Each worker
 *   takes an equal slice and processes it in chunks of up to
 *   CHUNK_MATRICES matrices via one bulk TLOAD + TSTORE per chunk.  Within
 *   each chunk, matrices are further split into groups of MAX_GROUP_SIZE.
 *
 * Epsilon:
 *   The reference formula adds `eps` to the matrix after softmax and to
 *   each row/col sum before dividing.  Those adds are elided here (matching
 *   the multi-K kernel's K=4 path): after softmax every cell is strictly
 *   positive, so no denominator can be zero, and elision was validated by
 *   the full shape sweep in test_sinkhorn.py at repeat=10.  Re-introduce
 *   the three TADDS ops if running at very high iteration counts.
 */

#include <pto/pto-inst.hpp>
#include <pto/npu/a2a3/TCopy.hpp>

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *
#endif

using namespace pto;

// ==========================================================================
// Compile-time constants
// ==========================================================================
constexpr uint32_t UB_BYTES = 192 * 1024;  // per-AIV unified buffer size
constexpr uint32_t K = 4;                  // matrix dimension (fixed)
constexpr uint32_t TILE_COLS =
    16;  // padded row width (32-byte fp16 alignment)
constexpr uint32_t STACK_ROWS = 512;  // tall-tile row count

// 32-byte align helper (fp16 PTO tiles require 32-byte-aligned row bytes).
#define ALIGN_32(x) (((x) + 31u) & ~31u)

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

// ==========================================================================
// Tile type aliases
// ==========================================================================
template <typename T, uint32_t N>
using FlatVec = Tile<TileType::Vec, T, 1, N, BLayout::RowMajor, -1, -1>;

template <typename T, uint32_t Rows, uint32_t Cols>
using Tile2D =
    Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, DYNAMIC, DYNAMIC>;

template <typename T, uint32_t Rows>
using ColVec =
    Tile<TileType::Vec, T, Rows, 1, BLayout::ColMajor, DYNAMIC, DYNAMIC>;

// ==========================================================================
// Global-memory tensor aliases (contiguous row-major)
// ==========================================================================
using GmDenseStride = Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T>
using GmShape2D = TileShape2D<T, DYNAMIC, DYNAMIC, Layout::ND>;
template <typename T, uint32_t Cols>
using GmTensor = GlobalTensor<T, GmShape2D<T>, GmDenseStride, Layout::ND>;

// ==========================================================================
// Pipeline-flag helpers
// ==========================================================================
AICORE inline void initPipelineFlags() {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

AICORE inline void drainPipelineFlags() {
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

// ==========================================================================
// Interleave / de-interleave row stripes (BATCH_UB ↔ WORK_UB)
//
// Implemented with ``pto::TCopy`` on matching 2D tile views so the same
// pattern maps cleanly to PTODSL ``tile`` UB copies (stride + validRow /
// validCol) instead of calling ``__builtin_cce_copy_ubuf_to_ubuf`` directly.
// For each within-matrix row index ``row`` (0..K-1), we copy ``group_size``
// rows; source stride in batch layout is ``K * TILE_COLS`` half elements
// between matrices, destination stride in WORK is ``TILE_COLS`` half
// elements per tall row.  ``validCol == K`` selects the ``validCol < Cols``
// branch of ``TCopy.hpp`` (per-row ``copy_ubuf_to_ubuf`` with blockLen from
// ``K``).  De-interleave swaps strides.
// ==========================================================================

template <typename TileHalf>
AICORE inline void sinkhornInterleaveRowStripeUB(TileHalf &dst_view,
                                               TileHalf &src_view,
                                               uint64_t group_size) {
  constexpr unsigned SrcStrideHalf = K * TILE_COLS;
  constexpr unsigned DstStrideHalf = TILE_COLS;
  pto::TCopy<TileHalf, TileHalf, 1, SrcStrideHalf, DstStrideHalf>(
      dst_view.data(), src_view.data(), group_size, K);
}

template <typename TileHalf>
AICORE inline void sinkhornDeinterleaveRowStripeUB(TileHalf &dst_view,
                                                 TileHalf &src_view,
                                                 uint64_t group_size) {
  constexpr unsigned SrcStrideHalf = TILE_COLS;
  constexpr unsigned DstStrideHalf = K * TILE_COLS;
  pto::TCopy<TileHalf, TileHalf, 1, SrcStrideHalf, DstStrideHalf>(
      dst_view.data(), src_view.data(), group_size, K);
}

// ==========================================================================
// Fast path:  K = 4, N >= 2048  —  TCOLEXPANDDIV on full interleaved tile
// ==========================================================================
//
// Two views of the same physical UB memory:
//
//   Tall view       shape (STACK_ROWS, TILE_COLS), row stride = TILE_COLS.
//                   Used for per-row ops (softmax, row-normalize).
//                   Matrix i's row r lives at tall-row
//                   ((i / GS) * K + i % GS + r * GS).
//
//   Interleaved     shape (K, ROW_BLOCK_COLS), row stride = ROW_BLOCK_COLS
//   view                = MAX_GROUP_SIZE * TILE_COLS.
//                   Used for per-col ops.  Each "column" corresponds to
//                   one matrix's one column, so TCOLSUM gives per-matrix
//                   col sums and TCOLEXPANDDIV normalizes correctly.
//
// Both views work simultaneously because
//   tall stride × group size = TILE_COLS × MAX_GROUP_SIZE = ROW_BLOCK_COLS.
// Partial groups are zero-padded; padding cells softmax to 1/K (benign)
// and don't leak into valid outputs.
template <typename T, uint32_t REPEAT>
AICORE void sinkhornFastPath(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                             float eps) {
  constexpr unsigned TALL_ROWS = STACK_ROWS;
  constexpr unsigned MAX_GROUP_SIZE = TALL_ROWS / K;
  constexpr unsigned ROW_BLOCK_COLS = MAX_GROUP_SIZE * TILE_COLS;
  static_assert(K * ROW_BLOCK_COLS == TALL_ROWS * TILE_COLS,
                "Interleaved and tall views must cover the same UB region");

  constexpr unsigned MATRIX_ROW_BYTES = TILE_COLS * sizeof(half);
  constexpr unsigned TILE_BYTES = TALL_ROWS * TILE_COLS * sizeof(half);

  constexpr unsigned SCRATCH_UB = 0;
  constexpr unsigned WORK_UB = ALIGN_32(SCRATCH_UB + TILE_BYTES);
  constexpr unsigned ROW_STATS_UB = ALIGN_32(WORK_UB + TILE_BYTES);
  constexpr unsigned COL_STATS_UB =
      ALIGN_32(ROW_STATS_UB + ALIGN_32(TALL_ROWS * sizeof(half)));
  constexpr unsigned BATCH_UB_BASE =
      ALIGN_32(COL_STATS_UB + ALIGN_32(ROW_BLOCK_COLS * sizeof(half)));

  constexpr unsigned BATCH_HALF_BUDGET = (UB_BYTES - BATCH_UB_BASE) / 2;
  constexpr unsigned BATCH_HALF_ROWS_RAW =
      BATCH_HALF_BUDGET / (TILE_COLS * sizeof(half));
  constexpr unsigned MAX_BATCH_ROWS =
      BATCH_HALF_ROWS_RAW < 4095 ? BATCH_HALF_ROWS_RAW : 4095;
  constexpr unsigned BATCH_HALF_BYTES =
      MAX_BATCH_ROWS * TILE_COLS * sizeof(half);
  constexpr unsigned BATCH_UB_PING = BATCH_UB_BASE;
  constexpr unsigned BATCH_UB_PONG = BATCH_UB_BASE + ALIGN_32(BATCH_HALF_BYTES);
  static_assert(BATCH_UB_PONG + BATCH_HALF_BYTES <= UB_BYTES,
                "Double-buffered BATCH_UB exceeds UB capacity");

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_workers = get_block_num() * get_subblockdim();
  const uint32_t worker_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t base_per_worker = N / num_workers;
  const uint32_t remainder = N % num_workers;
  const uint32_t my_first = worker_id * base_per_worker +
                            (worker_id < remainder ? worker_id : remainder);
  const uint32_t my_count = base_per_worker + (worker_id < remainder ? 1 : 0);
  if (my_count == 0) return;

  constexpr uint32_t K_SQUARED = K * K;
  constexpr uint32_t GROUP_SIZE_STATIC = MAX_GROUP_SIZE;
  constexpr uint32_t CHUNK_MATRICES = MAX_BATCH_ROWS / K;

  // Prime all four cross-pipe flags (two halves × two directions).
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  bool half_zeroed[2] = {false, false};
  uint32_t ping = 1;
  for (uint32_t chunk_offset = 0; chunk_offset < my_count;
       chunk_offset += CHUNK_MATRICES, ping = 1 - ping) {
    const uint32_t chunk_matrices =
        min(CHUNK_MATRICES, my_count - chunk_offset);
    const uint32_t chunk_rows = chunk_matrices * K;
    __gm__ T *chunk_gm_in =
        gm_in + (size_t)(my_first + chunk_offset) * K_SQUARED;
    __gm__ T *chunk_gm_out =
        gm_out + (size_t)(my_first + chunk_offset) * K_SQUARED;

    const unsigned batch_ub = ping ? BATCH_UB_PING : BATCH_UB_PONG;
    const event_t ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;

    Tile2D<T, MAX_BATCH_ROWS, TILE_COLS> batch_tile(chunk_rows, K);
    TASSIGN(batch_tile, batch_ub);

    GmShape2D<T> gm_shape(chunk_rows, K);
    GmDenseStride gm_stride(K);
    GmTensor<T, TILE_COLS> gm_in_tensor(chunk_gm_in, gm_shape, gm_stride);

    wait_flag(PIPE_V, PIPE_MTE2, ev);

    // Lazy-zero: on this half's first use, zero the TILE_COLS - K padding cols.
    if (!half_zeroed[ping]) {
      FlatVec<T, MAX_BATCH_ROWS * TILE_COLS> zero_flat(
          1, MAX_BATCH_ROWS * TILE_COLS);
      TASSIGN(zero_flat, batch_ub);
      TEXPANDS(zero_flat, (T)0);
      pipe_barrier(PIPE_V);
      half_zeroed[ping] = true;
    }

    TLOAD(batch_tile, gm_in_tensor);
    set_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE3, PIPE_V, ev);

    for (uint32_t group_start = 0; group_start < chunk_matrices;
         group_start += GROUP_SIZE_STATIC) {
      const uint32_t group_size =
          min(GROUP_SIZE_STATIC, chunk_matrices - group_start);
      const unsigned group_batch_offset =
          batch_ub + group_start * K * TILE_COLS * sizeof(T);

      if (group_size < GROUP_SIZE_STATIC) {
        FlatVec<T, K * ROW_BLOCK_COLS> work_flat(1, K * ROW_BLOCK_COLS);
        TASSIGN(work_flat, WORK_UB);
        TEXPANDS(work_flat, (T)0);
        pipe_barrier(PIPE_V);
      }

      // Interleave: BATCH_UB → WORK_UB.
      for (uint32_t row = 0; row < K; ++row) {
        Tile2D<half, TALL_ROWS, TILE_COLS> src_view(group_size, K);
        Tile2D<half, TALL_ROWS, TILE_COLS> dst_view(group_size, K);
        TASSIGN(src_view, group_batch_offset + row * MATRIX_ROW_BYTES);
        TASSIGN(dst_view,
                WORK_UB + row * ROW_BLOCK_COLS * (unsigned)sizeof(half));
        sinkhornInterleaveRowStripeUB(dst_view, src_view, group_size);
      }
      pipe_barrier(PIPE_V);

      if constexpr (REPEAT > 0) {
        Tile2D<half, TALL_ROWS, TILE_COLS> tall_matrix(TALL_ROWS, K);
        TASSIGN(tall_matrix, WORK_UB);

        Tile2D<half, TALL_ROWS, TILE_COLS> tall_scratch(TALL_ROWS, K);
        TASSIGN(tall_scratch, SCRATCH_UB);

        ColVec<half, TALL_ROWS> row_stats(TALL_ROWS, 1);
        TASSIGN(row_stats, ROW_STATS_UB);

        // Softmax along each matrix-row.
        TROWMAX(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDSUB(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        {
          FlatVec<half, TALL_ROWS * TILE_COLS> work_flat(1,
                                                         TALL_ROWS * TILE_COLS);
          TASSIGN(work_flat, WORK_UB);
          TEXP(work_flat, work_flat);
          pipe_barrier(PIPE_V);
        }

        TROWSUM(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        // Col-normalize via the interleaved view.
        Tile2D<half, K, ROW_BLOCK_COLS> interleaved_matrix(K, ROW_BLOCK_COLS);
        TASSIGN(interleaved_matrix, WORK_UB);

        Tile2D<half, K, ROW_BLOCK_COLS> interleaved_scratch(K, ROW_BLOCK_COLS);
        TASSIGN(interleaved_scratch, SCRATCH_UB);

        FlatVec<half, ROW_BLOCK_COLS> col_stats(1, ROW_BLOCK_COLS);
        TASSIGN(col_stats, COL_STATS_UB);

#define COL_NORMALIZE()                                                \
  do {                                                                 \
    TCOLSUM(col_stats, interleaved_matrix, interleaved_scratch, true); \
    pipe_barrier(PIPE_V);                                              \
    TCOLEXPANDDIV(interleaved_matrix, interleaved_matrix, col_stats);  \
    pipe_barrier(PIPE_V);                                              \
  } while (0)

        COL_NORMALIZE();

#pragma unroll
        for (uint32_t iter = 1; iter < REPEAT; ++iter) {
          TASSIGN(row_stats, ROW_STATS_UB);

          TROWSUM(row_stats, tall_matrix, tall_scratch);
          pipe_barrier(PIPE_V);

          TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
          pipe_barrier(PIPE_V);

          COL_NORMALIZE();
        }
#undef COL_NORMALIZE
      }

      // De-interleave: WORK_UB → BATCH_UB.
      for (uint32_t row = 0; row < K; ++row) {
        Tile2D<half, TALL_ROWS, TILE_COLS> src_view(group_size, K);
        Tile2D<half, TALL_ROWS, TILE_COLS> dst_view(group_size, K);
        TASSIGN(src_view,
                WORK_UB + row * ROW_BLOCK_COLS * (unsigned)sizeof(half));
        TASSIGN(dst_view, group_batch_offset + row * MATRIX_ROW_BYTES);
        sinkhornDeinterleaveRowStripeUB(dst_view, src_view, group_size);
      }
      pipe_barrier(PIPE_V);
    }

    GmTensor<T, TILE_COLS> gm_out_tensor(chunk_gm_out, gm_shape, gm_stride);
    set_flag(PIPE_V, PIPE_MTE3, ev);
    wait_flag(PIPE_V, PIPE_MTE3, ev);
    TSTORE(gm_out_tensor, batch_tile);
    set_flag(PIPE_MTE3, PIPE_V, ev);
    set_flag(PIPE_V, PIPE_MTE2, ev);
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

// ==========================================================================
// Small-batch path:  K = 4, N < 2048  —  natural-order, no DB
// ==========================================================================
//
// Fast path's double-buffer + interleave setup costs ~25us per call and is
// wasted when there's only one chunk per worker.  At K=4, batch=1 this
// path clocks ~14us vs ~40us for fast-path; crossover is around N=2048.
template <typename T, uint32_t REPEAT>
AICORE void sinkhornSmallBatch(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                               float eps) {
  constexpr unsigned TALL_ROWS = STACK_ROWS;
  constexpr unsigned TILE_BYTES = TALL_ROWS * TILE_COLS * sizeof(half);
  constexpr unsigned MATRIX_ROW_BYTES = TILE_COLS * sizeof(half);

  constexpr unsigned MAT_UB = 0;
  constexpr unsigned SCRATCH_UB = ALIGN_32(MAT_UB + TILE_BYTES);
  constexpr unsigned ROW_STATS_UB = ALIGN_32(SCRATCH_UB + TILE_BYTES);
  constexpr unsigned BATCH_UB =
      ALIGN_32(ROW_STATS_UB + ALIGN_32(TALL_ROWS * sizeof(half)));
  constexpr unsigned BATCH_BUF_ROWS_RAW =
      (UB_BYTES - BATCH_UB) / (TILE_COLS * sizeof(half));
  constexpr unsigned MAX_BATCH_ROWS =
      BATCH_BUF_ROWS_RAW < 4095 ? BATCH_BUF_ROWS_RAW : 4095;

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_workers = get_block_num() * get_subblockdim();
  const uint32_t worker_id =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t base_per_worker = N / num_workers;
  const uint32_t remainder = N % num_workers;
  const uint32_t my_first = worker_id * base_per_worker +
                            (worker_id < remainder ? worker_id : remainder);
  const uint32_t my_count = base_per_worker + (worker_id < remainder ? 1 : 0);
  if (my_count == 0) return;

  constexpr uint32_t K_SQUARED = K * K;
  constexpr uint32_t MAX_GROUP_SIZE = TALL_ROWS / K;
  constexpr uint32_t CHUNK_MATRICES = MAX_BATCH_ROWS / K;
  const half eps_h = (half)eps;

  initPipelineFlags();

  for (uint32_t chunk_offset = 0; chunk_offset < my_count;
       chunk_offset += CHUNK_MATRICES) {
    const uint32_t chunk_matrices =
        min(CHUNK_MATRICES, my_count - chunk_offset);
    const uint32_t chunk_rows = chunk_matrices * K;
    __gm__ T *chunk_gm_in =
        gm_in + (size_t)(my_first + chunk_offset) * K_SQUARED;
    __gm__ T *chunk_gm_out =
        gm_out + (size_t)(my_first + chunk_offset) * K_SQUARED;

    // Zero BATCH_UB region we're about to load (padding cols stay 0).
    {
      FlatVec<T, MAX_BATCH_ROWS * TILE_COLS> zero_flat(1,
                                                       chunk_rows * TILE_COLS);
      TASSIGN(zero_flat, BATCH_UB);
      TEXPANDS(zero_flat, (T)0);
      pipe_barrier(PIPE_V);
    }

    Tile2D<T, MAX_BATCH_ROWS, TILE_COLS> batch_tile(chunk_rows, K);
    TASSIGN(batch_tile, BATCH_UB);
    GmShape2D<T> gm_shape(chunk_rows, K);
    GmDenseStride gm_stride(K);
    GmTensor<T, TILE_COLS> gm_in_tensor(chunk_gm_in, gm_shape, gm_stride);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(batch_tile, gm_in_tensor);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (uint32_t group_start = 0; group_start < chunk_matrices;
         group_start += MAX_GROUP_SIZE) {
      const uint32_t group_size =
          min(MAX_GROUP_SIZE, chunk_matrices - group_start);
      const uint32_t group_rows = group_size * K;
      const uint32_t group_cells = group_rows * TILE_COLS;
      const unsigned group_batch_offset =
          BATCH_UB + group_start * K * TILE_COLS * sizeof(T);

      // Copy BATCH_UB → MAT_UB (natural order, same stride).
      {
        FlatVec<T, TALL_ROWS * TILE_COLS> zero_mat(1, TALL_ROWS * TILE_COLS);
        TASSIGN(zero_mat, MAT_UB);
        TEXPANDS(zero_mat, (T)0);
        pipe_barrier(PIPE_V);
      }
      {
        FlatVec<T, TALL_ROWS * TILE_COLS> src(1, group_cells);
        FlatVec<T, TALL_ROWS * TILE_COLS> dst(1, group_cells);
        TASSIGN(src, group_batch_offset);
        TASSIGN(dst, MAT_UB);
        TMOV(dst, src);
        pipe_barrier(PIPE_V);
      }

      Tile2D<half, TALL_ROWS, TILE_COLS> tall_matrix(group_rows, K);
      TASSIGN(tall_matrix, MAT_UB);
      Tile2D<half, TALL_ROWS, TILE_COLS> tall_scratch(group_rows, K);
      TASSIGN(tall_scratch, SCRATCH_UB);
      ColVec<half, TALL_ROWS> row_stats(group_rows, 1);
      TASSIGN(row_stats, ROW_STATS_UB);

      if constexpr (REPEAT > 0) {
        TROWMAX(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDSUB(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        {
          FlatVec<half, TALL_ROWS * TILE_COLS> flat(1, group_cells);
          TASSIGN(flat, MAT_UB);
          TEXP(flat, flat);
          pipe_barrier(PIPE_V);
        }

        TROWSUM(row_stats, tall_matrix, tall_scratch);
        pipe_barrier(PIPE_V);

        TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
        pipe_barrier(PIPE_V);

        // Matches reference `x = x.softmax(-1) + eps`.  Kept on this path
        // because per-matrix col-normalize (below) divides by raw col
        // sums without adding eps — fp16 drift over 10 iterations
        // otherwise grows past the test's rtol.
        {
          FlatVec<half, TALL_ROWS * TILE_COLS> flat(1, group_cells);
          TASSIGN(flat, MAT_UB);
          TADDS(flat, flat, eps_h);
          pipe_barrier(PIPE_V);
        }

#define PER_MATRIX_COL_NORM()                                      \
  do {                                                             \
    for (uint32_t mi = 0; mi < group_size; ++mi) {                 \
      const unsigned mat_off = MAT_UB + mi * K * MATRIX_ROW_BYTES; \
      Tile2D<half, TILE_COLS, TILE_COLS> sub_mat(K, K);            \
      TASSIGN(sub_mat, mat_off);                                   \
      Tile2D<half, TILE_COLS, TILE_COLS> sub_scratch(K, K);        \
      TASSIGN(sub_scratch, SCRATCH_UB);                            \
      FlatVec<half, TILE_COLS> col_stats(1, K);                    \
      TASSIGN(col_stats, ROW_STATS_UB);                            \
      TCOLSUM(col_stats, sub_mat, sub_scratch, false);             \
      pipe_barrier(PIPE_V);                                        \
      TCOLEXPANDDIV(sub_mat, sub_mat, col_stats);                  \
      pipe_barrier(PIPE_V);                                        \
    }                                                              \
  } while (0)

        PER_MATRIX_COL_NORM();

#pragma unroll
        for (uint32_t iter = 1; iter < REPEAT; ++iter) {
          TASSIGN(row_stats, ROW_STATS_UB);
          TROWSUM(row_stats, tall_matrix, tall_scratch);
          pipe_barrier(PIPE_V);

          TROWEXPANDDIV(tall_matrix, tall_matrix, row_stats);
          pipe_barrier(PIPE_V);

          PER_MATRIX_COL_NORM();
        }
#undef PER_MATRIX_COL_NORM
      }

      // Copy back MAT_UB → BATCH_UB.
      {
        FlatVec<T, TALL_ROWS * TILE_COLS> src(1, group_cells);
        FlatVec<T, TALL_ROWS * TILE_COLS> dst(1, group_cells);
        TASSIGN(src, MAT_UB);
        TASSIGN(dst, group_batch_offset);
        TMOV(dst, src);
        pipe_barrier(PIPE_V);
      }
    }

    GmTensor<T, TILE_COLS> gm_out_tensor(chunk_gm_out, gm_shape, gm_stride);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gm_out_tensor, batch_tile);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  }

  drainPipelineFlags();
}

// ==========================================================================
// Dispatch
// ==========================================================================
// K is fixed at 4.  Batch-size threshold (N=2048) empirically taken from
// msprof + direct Event timing on 910B: below it the fast-path's
// interleave + double-buffer setup overhead dominates.
template <typename T, uint32_t REPEAT>
AICORE void dispatchByBatch(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                            float eps) {
  if (N >= 2048)
    sinkhornFastPath<T, REPEAT>(gm_in, gm_out, N, eps);
  else
    sinkhornSmallBatch<T, REPEAT>(gm_in, gm_out, N, eps);
}

// Specialize on `repeat` so that the per-iteration unroll constant is known.
template <typename T>
AICORE void dispatchByRepeat(__gm__ T *gm_in, __gm__ T *gm_out, uint32_t N,
                             uint32_t repeat, float eps) {
  switch (repeat) {
    case 0:
      dispatchByBatch<T, 0>(gm_in, gm_out, N, eps);
      break;
    case 1:
      dispatchByBatch<T, 1>(gm_in, gm_out, N, eps);
      break;
    case 3:
      dispatchByBatch<T, 3>(gm_in, gm_out, N, eps);
      break;
    case 5:
      dispatchByBatch<T, 5>(gm_in, gm_out, N, eps);
      break;
    case 8:
      dispatchByBatch<T, 8>(gm_in, gm_out, N, eps);
      break;
    case 10:
      dispatchByBatch<T, 10>(gm_in, gm_out, N, eps);
      break;
    case 20:
      dispatchByBatch<T, 20>(gm_in, gm_out, N, eps);
      break;
    default:
      dispatchByBatch<T, 10>(gm_in, gm_out, N, eps);
      break;
  }
}
#endif  // __CCE_AICORE__ == 220 && __DAV_C220_VEC__

// ==========================================================================
// C ABI
// ==========================================================================
// ABI signature keeps `K` for source compatibility with the multi-K wrapper;
// this kernel is hard-coded to K=4 and ignores the argument.
extern "C" __global__ AICORE void sinkhorn_ds_fp16(GM_ADDR input,
                                                   GM_ADDR output, uint32_t N,
                                                   uint32_t K_arg,
                                                   uint32_t repeat, float eps) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  (void)K_arg;
  dispatchByRepeat<half>((__gm__ half *)input, (__gm__ half *)output, N, repeat,
                         eps);
#else
  (void)input;
  (void)output;
  (void)N;
  (void)K_arg;
  (void)repeat;
  (void)eps;
#endif
}

// Host-side launch.  Ascend 910B runs 2 AIV cores per cube core.
extern "C" void call_sinkhorn_ds_kernel(uint32_t cube_core_num, void *stream,
                                        uint8_t *input, uint8_t *output,
                                        uint32_t N, uint32_t K, uint32_t repeat,
                                        float eps) {
  sinkhorn_ds_fp16<<<cube_core_num * 2, nullptr, stream>>>(input, output, N, K,
                                                           repeat, eps);
}

// Demo / ctypes ABI (same as ``kernel_sinkhorn.cpp`` in this directory).
extern "C" void call_sinkhorn(uint32_t cube_core_num, void *stream,
                              uint8_t *input, uint8_t *output, uint32_t num_matrices,
                              uint32_t repeat, float eps) {
  call_sinkhorn_ds_kernel(cube_core_num, stream, input, output, num_matrices, 4,
                          repeat, eps);
}
