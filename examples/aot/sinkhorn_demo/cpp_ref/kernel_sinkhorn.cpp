/**
 * Doubly-stochastic Sinkhorn normalization — minimal PTO demo (fp16, K=4).
 *
 * Mirrors DeepSeek TileKernels `sinkhorn_normalize_ref`:
 *
 *     x = softmax(x, -1) + eps
 *     x = x / (colsum(x) + eps)                        # first col normalize
 *     for _ in range(repeat - 1):
 *         x = x / (rowsum(x) + eps)                    # row normalize
 *         x = x / (colsum(x) + eps)                    # col normalize
 *
 * Parallelism & batching:
 *   Each AIV core processes BATCH matrices at a time (one TLOAD for the
 *   whole group, one TSTORE at the end). The BATCH matrices are stacked
 *   vertically in UB to form a (BATCH·K)×TILE_DIM tile. This lets us run
 *   per-matrix-row ops (softmax, row-normalize) as a SINGLE call over the
 *   whole stack. Column-normalize can't batch the same way (a TCOLSUM
 *   over the stack would mix columns across matrices), so that step
 *   loops over the BATCH matrices as independent K×K sub-tiles.
 *
 * Fp16 PTO tiles must have row-bytes that are a multiple of 32, i.e. the
 * tile column dim must be a multiple of 16. K=4 is smaller than that, so
 * each matrix's row is padded out to 16; the padding cells stay at 0 for
 * the whole computation.
 */

#include <pto/pto-inst.hpp>

using namespace pto;

// ---- Problem constants ----
constexpr uint32_t K         = 4;             // matrix dimension
constexpr uint32_t TILE_DIM  = 16;            // padded tile col dim (32-byte row alignment for fp16)
constexpr uint32_t BATCH     = 8;             // matrices processed together per AIV core
constexpr uint32_t STACK_ROWS = BATCH * K;    // rows of the stacked-matrix tile

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

// ---- Tile type aliases ----
// The stacked BATCH×(K×K) matrices as one (BATCH·K, TILE_DIM) tile.
using Stack = Tile<TileType::Vec, half, STACK_ROWS, TILE_DIM,
                   BLayout::RowMajor, DYNAMIC, DYNAMIC>;

// 1-D view over the stack (used when we only need elementwise ops on it).
using StackFlat = Tile<TileType::Vec, half, 1, STACK_ROWS * TILE_DIM,
                       BLayout::RowMajor, -1, -1>;

// Holds one scalar per row of the stack (= one per matrix-row).
using StackRowStat = Tile<TileType::Vec, half, STACK_ROWS, 1,
                          BLayout::ColMajor, DYNAMIC, DYNAMIC>;

// One K×K matrix view (for the per-matrix col-normalize loop).
using SubMatrix = Tile<TileType::Vec, half, TILE_DIM, TILE_DIM,
                       BLayout::RowMajor, DYNAMIC, DYNAMIC>;

// Holds one scalar per column of a sub-matrix.
using SubColStat = Tile<TileType::Vec, half, 1, TILE_DIM,
                        BLayout::RowMajor, -1, -1>;

// ---- Global-memory view aliases ----
using GmStride = Stride<1, 1, 1, DYNAMIC, 1>;
using GmShape  = TileShape2D<half, DYNAMIC, DYNAMIC, Layout::ND>;
using GmTensor = GlobalTensor<half, GmShape, GmStride, Layout::ND>;


AICORE void sinkhornK4(__gm__ half *in, __gm__ half *out,
                       uint32_t num_matrices, uint32_t repeat, float eps) {
  // UB memory layout — three (BATCH·K)×TILE_DIM stacks + one vector slot.
  constexpr unsigned STACK_BYTES  = STACK_ROWS * TILE_DIM * sizeof(half);
  constexpr unsigned MATRIX_BUF   = 0;
  constexpr unsigned SCRATCH_BUF  = MATRIX_BUF  + STACK_BYTES;
  constexpr unsigned ROW_STAT_BUF = SCRATCH_BUF + STACK_BYTES;                       // STACK_ROWS halves
  constexpr unsigned COL_STAT_BUF = ROW_STAT_BUF + STACK_ROWS * sizeof(half);        // TILE_DIM halves

  set_mask_norm();
  set_vector_mask(-1, -1);

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t core_id   = get_block_idx() * get_subblockdim() + get_subblockid();
  const half     eps_h     = (half)eps;

  // Initial cross-pipe flags — prime them so the first wait_flag below succeeds.
  set_flag(PIPE_V,    PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V,    EVENT_ID0);

  // Each AIV core processes a BATCH at a time; cores stride through the batches.
  for (uint32_t group = core_id * BATCH; group < num_matrices; group += num_cores * BATCH) {
    const uint32_t actual = min(BATCH, num_matrices - group);  // real matrices in this group (last may be partial)
    const uint32_t rows   = actual * K;

    __gm__ half *in_gm  = in  + (size_t)group * K * K;
    __gm__ half *out_gm = out + (size_t)group * K * K;

    // ── Load: zero the whole stack, then TLOAD the `rows` valid matrix-rows ──
    {
      StackFlat zeros(1, STACK_ROWS * TILE_DIM);
      TASSIGN(zeros, MATRIX_BUF);
      TEXPANDS(zeros, (half)0.f);
      pipe_barrier(PIPE_V);
    }

    Stack mat(rows, K);
    TASSIGN(mat, MATRIX_BUF);

    GmShape  gm_shape(rows, K);
    GmStride gm_stride(K);
    GmTensor gm_in(in_gm, gm_shape, gm_stride);

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(mat, gm_in);
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Reusable views of the scratch + row-stat buffers.
    Stack scratch(rows, K);
    TASSIGN(scratch, SCRATCH_BUF);

    StackRowStat row_stat(rows, 1);
    TASSIGN(row_stat, ROW_STAT_BUF);

    // ── Step 1: softmax along the last dim (BATCHED — one call for all matrices) ──
    //   row_stat[i] = max(mat[i, :])          (K=4 scalars per matrix × BATCH matrices)
    //   mat[i, :]   = exp(mat[i, :] - row_stat[i])
    //   row_stat[i] = sum(mat[i, :])
    //   mat[i, :]   = mat[i, :] / row_stat[i]
    TROWMAX(row_stat, mat, scratch);
    pipe_barrier(PIPE_V);

    TROWEXPANDSUB(mat, mat, row_stat);
    pipe_barrier(PIPE_V);

    {
      StackFlat flat(1, rows * TILE_DIM);
      TASSIGN(flat, MATRIX_BUF);
      TEXP(flat, flat);
      pipe_barrier(PIPE_V);
    }

    TROWSUM(row_stat, mat, scratch);
    pipe_barrier(PIPE_V);

    TROWEXPANDDIV(mat, mat, row_stat);
    pipe_barrier(PIPE_V);

    // ── Step 2: mat += eps  (batched) ───────────────────────────────────
    {
      StackFlat flat(1, rows * TILE_DIM);
      TASSIGN(flat, MATRIX_BUF);
      TADDS(flat, flat, eps_h);
      pipe_barrier(PIPE_V);
    }

    // ── Step 3: first col-normalize — per-matrix loop ───────────────────
    //   Column-normalize can't batch across stacked matrices (TCOLSUM on
    //   the stack would mix cols across matrices), so we loop once over
    //   each K×K sub-matrix.
    for (uint32_t m = 0; m < actual; ++m) {
      const unsigned offset = MATRIX_BUF + m * K * TILE_DIM * sizeof(half);

      SubMatrix  sub_mat(K, K);       TASSIGN(sub_mat,  offset);
      SubMatrix  sub_scratch(K, K);   TASSIGN(sub_scratch, SCRATCH_BUF + m * K * TILE_DIM * sizeof(half));
      SubColStat col_stat(1, K);      TASSIGN(col_stat, COL_STAT_BUF);

      TCOLSUM(col_stat, sub_mat, sub_scratch, false);
      pipe_barrier(PIPE_V);

      TADDS(col_stat, col_stat, eps_h);
      pipe_barrier(PIPE_V);

      TCOLEXPANDDIV(sub_mat, sub_mat, col_stat);
      pipe_barrier(PIPE_V);
    }

    // ── Step 4: (repeat - 1) × { batched row-normalize; per-matrix col-normalize } ──
    for (uint32_t iter = 1; iter < repeat; ++iter) {
      // Batched row-normalize.
      TASSIGN(row_stat, ROW_STAT_BUF);

      TROWSUM(row_stat, mat, scratch);
      pipe_barrier(PIPE_V);

      TADDS(row_stat, row_stat, eps_h);
      pipe_barrier(PIPE_V);

      TROWEXPANDDIV(mat, mat, row_stat);
      pipe_barrier(PIPE_V);

      // Per-matrix col-normalize.
      for (uint32_t m = 0; m < actual; ++m) {
        const unsigned offset = MATRIX_BUF + m * K * TILE_DIM * sizeof(half);

        SubMatrix  sub_mat(K, K);      TASSIGN(sub_mat,  offset);
        SubMatrix  sub_scratch(K, K);  TASSIGN(sub_scratch, SCRATCH_BUF + m * K * TILE_DIM * sizeof(half));
        SubColStat col_stat(1, K);     TASSIGN(col_stat, COL_STAT_BUF);

        TCOLSUM(col_stat, sub_mat, sub_scratch, false);
        pipe_barrier(PIPE_V);

        TADDS(col_stat, col_stat, eps_h);
        pipe_barrier(PIPE_V);

        TCOLEXPANDDIV(sub_mat, sub_mat, col_stat);
        pipe_barrier(PIPE_V);
      }
    }

    // ── Store the valid rows back to GM ─────────────────────────────────
    GmTensor gm_out(out_gm, gm_shape, gm_stride);

    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(gm_out, mat);
    pipe_barrier(PIPE_ALL);

    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_V,    PIPE_MTE2, EVENT_ID0);
  }

  // Drain pipelines before exit.
  wait_flag(PIPE_V,    PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V,    EVENT_ID0);
}
#endif  // __CCE_AICORE__ == 220 && __DAV_C220_VEC__


// ---- C ABI ---------------------------------------------------------------

extern "C" __global__ AICORE void sinkhorn_k4_fp16(
    __gm__ uint8_t *input, __gm__ uint8_t *output,
    uint32_t num_matrices, uint32_t repeat, float eps) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  sinkhornK4((__gm__ half *)input, (__gm__ half *)output,
             num_matrices, repeat, eps);
#else
  (void)input;
  (void)output;
  (void)num_matrices;
  (void)repeat;
  (void)eps;
#endif
}

// Host-side launch. Ascend 910B runs 2 AIV cores per cube core.
extern "C" void call_sinkhorn(
    uint32_t cube_core_num, void *stream,
    uint8_t *input, uint8_t *output,
    uint32_t num_matrices, uint32_t repeat, float eps) {
  sinkhorn_k4_fp16<<<cube_core_num * 2, nullptr, stream>>>(
      input, output, num_matrices, repeat, eps);
}
