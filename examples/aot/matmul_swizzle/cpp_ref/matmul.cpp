#include <pto/pto-inst.hpp>

using namespace pto;

namespace detail {

// ---------------------------------------------------------------------------
// Flag helpers (same pattern as block_rotate reference)
// ---------------------------------------------------------------------------
template <pipe_t Src, pipe_t Dst>
AICORE inline void SetFlag(uint32_t id) {
  set_flag(Src, Dst, static_cast<event_t>(id));
}
template <pipe_t Src, pipe_t Dst>
AICORE inline void WaitFlag(uint32_t id) {
  wait_flag(Src, Dst, static_cast<event_t>(id));
}

// ---------------------------------------------------------------------------
// L2 swizzle — remap (midx, nidx) for cache locality
// ---------------------------------------------------------------------------
AICORE inline void SwizzleBlockIdx(int32_t loop_idx, int32_t m_loop,
                                   int32_t n_loop, int32_t swizzle_direction,
                                   int32_t swizzle_count, int64_t& m_idx,
                                   int64_t& n_idx) {
  m_idx = loop_idx / n_loop;
  n_idx = loop_idx % n_loop;

  if (swizzle_count <= 0) return;
  if (swizzle_direction != 0 && swizzle_direction != 1) return;

  uint32_t in_batch_index = loop_idx % (m_loop * n_loop);
  if (swizzle_direction == 0) {  // Zn
    uint32_t tile_block_loop = (m_loop + swizzle_count - 1) / swizzle_count;
    uint32_t tile_block_idx = in_batch_index / (swizzle_count * n_loop);
    uint32_t in_tile_block_idx = in_batch_index % (swizzle_count * n_loop);
    uint32_t n_row = (tile_block_idx == tile_block_loop - 1)
                         ? (m_loop - swizzle_count * tile_block_idx)
                         : swizzle_count;
    m_idx = tile_block_idx * swizzle_count + in_tile_block_idx % n_row;
    n_idx = in_tile_block_idx / n_row;
    if (tile_block_idx % 2 != 0) n_idx = n_loop - n_idx - 1;
  } else if (swizzle_direction == 1) {  // Nz
    uint32_t tile_block_loop = (n_loop + swizzle_count - 1) / swizzle_count;
    uint32_t tile_block_idx = in_batch_index / (swizzle_count * m_loop);
    uint32_t in_tile_block_idx = in_batch_index % (swizzle_count * m_loop);
    uint32_t n_col = (tile_block_idx == tile_block_loop - 1)
                         ? (n_loop - swizzle_count * tile_block_idx)
                         : swizzle_count;
    m_idx = in_tile_block_idx / n_col;
    n_idx = tile_block_idx * swizzle_count + in_tile_block_idx % n_col;
    if (tile_block_idx % 2 != 0) m_idx = m_loop - m_idx - 1;
  }
}

// ---------------------------------------------------------------------------
// Tiling constants
// ---------------------------------------------------------------------------
constexpr int M_TILE = 128;
constexpr int K_QTILE = 64;                // quarter-K (L0 compute granularity)
constexpr int K_TILE = 256;                // half-K    (L1 B load granularity)
constexpr int K_DTILE = 512;               // double-K  (L1 A load granularity)
constexpr int PHASES = K_DTILE / K_QTILE;  // 8 phases per kDtile

// L1 byte sizes for A buffers (independent of N_TILE)
constexpr uintptr_t A_FULL_BYTES = M_TILE * K_DTILE * sizeof(half);  // 128 KB
constexpr uintptr_t A_SUB_BYTES = M_TILE * K_QTILE * sizeof(half);   //  16 KB
constexpr uintptr_t B_L1_START = 2 * A_FULL_BYTES;                   // 256 KB

// ---------------------------------------------------------------------------
// Core tile computation — templated on N_TILE (256 or 128)
// ---------------------------------------------------------------------------
template <int N_TILE>
AICORE void computeTile(__gm__ half* x, __gm__ half* y, __gm__ half* z,
                        int mOffset, int nOffset, int K, int N, int kDtileNum,
                        int& curr, bool& firstIter) {
  // byte sizes that depend on N_TILE
  constexpr uintptr_t B_HALF_BYTES = N_TILE * K_TILE * sizeof(half);
  constexpr uintptr_t B_SUB_BYTES = N_TILE * K_QTILE * sizeof(half);

  // ---- L1 tile types (Mat tiles in NZ / fractal format) ----
  using L1ALarge = Tile<TileType::Mat, half, M_TILE, K_DTILE, BLayout::ColMajor,
                        M_TILE, K_DTILE, SLayout::RowMajor, 512>;
  // B is sourced as DN(K,N) from row-major (N,K) GM without host transpose.
  using L1BLarge = Tile<TileType::Mat, half, K_TILE, N_TILE, BLayout::RowMajor,
                        K_TILE, N_TILE, SLayout::ColMajor, 512>;

  // ---- L0 tile types ----
  using L0A = TileLeft<half, M_TILE, K_QTILE>;
  using L0B = TileRight<half, K_QTILE, N_TILE>;
  using L0C = TileAcc<float, M_TILE, N_TILE>;

  // ---- GlobalTensor types ----
  using DynStrideND = Stride<1, 1, 1, DYNAMIC, 1>;
  using DynStrideDN = Stride<1, 1, 1, 1, DYNAMIC>;
  using GlobA =
      GlobalTensor<half, TileShape2D<half, M_TILE, K_DTILE, Layout::ND>,
                   DynStrideND, Layout::ND>;
  using GlobB =
      GlobalTensor<half, TileShape2D<half, K_TILE, N_TILE, Layout::DN>,
                   DynStrideDN, Layout::DN>;
  using GlobC =
      GlobalTensor<half, TileShape2D<half, M_TILE, N_TILE, Layout::ND>,
                   DynStrideND, Layout::ND>;

  // ---- Allocate L1 buffers ----
  L1ALarge a_l1[2];
  TASSIGN(a_l1[0], static_cast<uintptr_t>(0));
  TASSIGN(a_l1[1], A_FULL_BYTES);

  L1BLarge b_l1[2];
  TASSIGN(b_l1[0], B_L1_START);
  TASSIGN(b_l1[1], B_L1_START + B_HALF_BYTES);

  // ---- Allocate L0 buffers ----
  L0A a_l0[2];
  TASSIGN(a_l0[0], static_cast<uintptr_t>(0));
  TASSIGN(a_l0[1], A_SUB_BYTES);

  L0B b_l0[2];
  TASSIGN(b_l0[0], static_cast<uintptr_t>(0));
  TASSIGN(b_l0[1], B_SUB_BYTES);

  L0C c_l0;
  TASSIGN(c_l0, static_cast<uintptr_t>(0));

  // ---- Wait for previous tile's TSTORE to finish ----
  if (!firstIter) WaitFlag<PIPE_FIX, PIPE_M>(0);

  // ---- Load first A tile (GM → L1 with ND→NZ conversion) ----
  WaitFlag<PIPE_MTE1, PIPE_MTE2>(curr);
  {
    GlobA ga(x + mOffset * K, {}, DynStrideND(K));
    TLOAD(a_l1[curr], ga);
  }
  SetFlag<PIPE_MTE2, PIPE_MTE1>(curr);

  // ---- K-dimension loop ----
  for (int kIdx = 0; kIdx < kDtileNum; ++kIdx) {
    const bool isFirstKTile = (kIdx == 0);
    const int next = 1 - curr;
    const int kOffset = kIdx * K_DTILE;

    // Process two B half-tiles per kDtile
    for (int h = 0; h < 2; ++h) {
      const int bEvt = 2 + h;  // EVENT_ID2 for h=0, EVENT_ID3 for h=1

      // Load B half-tile (GM → L1)
      WaitFlag<PIPE_MTE1, PIPE_MTE2>(bEvt);
      {
        GlobB gb(y + nOffset * K + kOffset + h * K_TILE, {}, DynStrideDN(K));
        TLOAD(b_l1[h], gb);
      }
      SetFlag<PIPE_MTE2, PIPE_MTE1>(bEvt);

      // 4 quarter-K phases per B half-tile
      for (int quarterPhaseIdx = 0; quarterPhaseIdx < 4; ++quarterPhaseIdx) {
        const int phaseIdx = h * 4 + quarterPhaseIdx;
        const int pingPongIdx = phaseIdx & 1;  // L0 ping-pong index
        const bool isFirstPhase = isFirstKTile && (phaseIdx == 0);

        // -- wait for M pipe to release L0[pingPongIdx] from 2 phases ago --
        WaitFlag<PIPE_M, PIPE_MTE1>(pingPongIdx);

        // -- wait for A data in L1 (only needed at phaseIdx 0) --
        if (phaseIdx == 0) WaitFlag<PIPE_MTE2, PIPE_MTE1>(curr);

        // -- Extract A sub-tile → L0A[pingPongIdx] --
        TEXTRACT(a_l0[pingPongIdx], a_l1[curr], 0, phaseIdx * K_QTILE);

        // release A L1 buffer after the last TEXTRACT from it (phaseIdx 7)
        if (phaseIdx == PHASES - 1) SetFlag<PIPE_MTE1, PIPE_MTE2>(curr);

        // -- wait for B data in L1 (only at start of each half) --
        if (quarterPhaseIdx == 0) WaitFlag<PIPE_MTE2, PIPE_MTE1>(bEvt);

        // -- Extract B sub-tile → L0B[pingPongIdx] --
        TEXTRACT(b_l0[pingPongIdx], b_l1[h], quarterPhaseIdx * K_QTILE, 0);

        // Keep MTE1->M on one event lane (id 0). Using id 1 can stall on some
        // setups.
        SetFlag<PIPE_MTE1, PIPE_M>(0);

        // release B L1 buffer after the last TEXTRACT from this half
        if (quarterPhaseIdx == 3) SetFlag<PIPE_MTE1, PIPE_MTE2>(bEvt);

        WaitFlag<PIPE_MTE1, PIPE_M>(0);

        if (isFirstPhase)
          TMATMUL(c_l0, a_l0[pingPongIdx], b_l0[pingPongIdx]);
        else
          TMATMUL_ACC(c_l0, c_l0, a_l0[pingPongIdx], b_l0[pingPongIdx]);

        // signal MTE1: L0[pingPongIdx] consumed, safe to overwrite
        SetFlag<PIPE_M, PIPE_MTE1>(pingPongIdx);
      }
    }

    // Prefetch next A tile (overlapped with computation of current)
    if (kIdx + 1 < kDtileNum) {
      WaitFlag<PIPE_MTE1, PIPE_MTE2>(next);
      {
        GlobA ga(x + mOffset * K + kOffset + K_DTILE, {}, DynStrideND(K));
        TLOAD(a_l1[next], ga);
      }
      SetFlag<PIPE_MTE2, PIPE_MTE1>(next);
    }
    curr ^= 1;
  }

  // ---- Store result (L0C → GM with automatic F32→F16 + NZ→ND) ----
  SetFlag<PIPE_M, PIPE_FIX>(0);
  WaitFlag<PIPE_M, PIPE_FIX>(0);
  {
    GlobC gc(z + mOffset * N + nOffset, {}, DynStrideND(N));
    TSTORE(gc, c_l0);
  }
  firstIter = false;
}

}  // namespace detail

// ===========================================================================
// Kernel entry point
// ===========================================================================
/**
 * @brief AICORE kernel for matrix multiplication with L1 & L0 double-buffering.
 * Computes C = A × B^T where tiles are distributed across cores via swizzling.
 * @param x Pointer to A matrix in global memory (M × K).
 * @param y Pointer to B matrix in global memory (N × K).
 * @param z Pointer to C output in global memory (M × N).
 * @param M Number of rows in A and C.
 * @param N Number of rows in B (columns of C).
 * @param K Number of columns in A and B.
 */
extern "C" __global__ AICORE void matmul_kernel_ABt(__gm__ uint8_t* x,
                                                    __gm__ uint8_t* y,
                                                    __gm__ uint8_t* z, int M,
                                                    int N, int K) {
#if defined(__DAV_CUBE__)
  __gm__ half* xh = (__gm__ half*)x;
  __gm__ half* yh = (__gm__ half*)y;
  __gm__ half* zh = (__gm__ half*)z;

  constexpr int N_FULL = 256;

  const int nLoop = (N + N_FULL - 1) / N_FULL;
  const int mLoop = M / detail::M_TILE;
  if (mLoop <= 0 || nLoop <= 0) return;

  constexpr int swizzle_direction = 1;  // Nz  (direction 0=Zn, 1=Nz)
  constexpr int swizzle_count = 3;

  const int coreLoop = nLoop * mLoop;
  const int kDtileNum = K / detail::K_DTILE;

  // ---- Initialise event flags ----
  detail::SetFlag<PIPE_M, PIPE_MTE1>(0);
  detail::SetFlag<PIPE_M, PIPE_MTE1>(1);
  detail::SetFlag<PIPE_MTE1, PIPE_MTE2>(0);
  detail::SetFlag<PIPE_MTE1, PIPE_MTE2>(1);
  detail::SetFlag<PIPE_MTE1, PIPE_MTE2>(2);
  detail::SetFlag<PIPE_MTE1, PIPE_MTE2>(3);

  int curr = 0;
  bool firstIter = true;
  for (int32_t li = 0; li < coreLoop; ++li) {
    if (li % block_num != get_block_idx()) continue;

    int64_t mi, ni;
    detail::SwizzleBlockIdx(li, mLoop, nLoop, swizzle_direction, swizzle_count,
                            mi, ni);

    const int mOffset = mi * detail::M_TILE;
    const int nOffset = ni * N_FULL;
    const int nTileSize = (nOffset + N_FULL > N) ? 128 : N_FULL;

    if (nTileSize == N_FULL)
      detail::computeTile<256>(xh, yh, zh, mOffset, nOffset, K, N, kDtileNum,
                               curr, firstIter);
    else
      detail::computeTile<128>(xh, yh, zh, mOffset, nOffset, K, N, kDtileNum,
                               curr, firstIter);

    // Allow FIX→M overlap between tiles
    if (li + block_num < coreLoop) detail::SetFlag<PIPE_FIX, PIPE_M>(0);
  }

  // ---- Drain all outstanding events ----
  detail::WaitFlag<PIPE_MTE1, PIPE_MTE2>(3);
  detail::WaitFlag<PIPE_MTE1, PIPE_MTE2>(2);
  detail::WaitFlag<PIPE_MTE1, PIPE_MTE2>(1);
  detail::WaitFlag<PIPE_MTE1, PIPE_MTE2>(0);
  detail::WaitFlag<PIPE_M, PIPE_MTE1>(0);
  detail::WaitFlag<PIPE_M, PIPE_MTE1>(1);
#endif  // __DAV_CUBE__
}

// ===========================================================================
// Host launcher
// ===========================================================================
/**
 * @brief Host launcher for matmul kernel.
 * @param blockDim          Number of kernel blocks (cores) to launch.
 * @param stream            Execution stream handle.
 * @param x                 Pointer to A matrix in global memory (M × K).
 * @param y                 Pointer to B matrix in global memory (N × K).
 * @param z                 Pointer to C output in global memory (M × N).
 * @param M                 Number of rows in A and C.
 * @param N                 Number of rows in B (columns of C).
 * @param K                 Number of columns in A and B.
 */
extern "C" void call_kernel(uint32_t blockDim, void* stream, uint8_t* x,
                            uint8_t* y, uint8_t* z, int M, int N, int K) {
  matmul_kernel_ABt<<<blockDim, nullptr, stream>>>(x, y, z, M, N, K);
}
