#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

constexpr uint32_t UB_ALLOC_BYTES = 32 * 1024;
constexpr uint32_t UB_HALF_BYTES = UB_ALLOC_BYTES / 2;
constexpr uint32_t ELEMENTS_PER_TILE = UB_ALLOC_BYTES / 2;  // half is 2 bytes

// Double-buffered UB memory layout (ping/pong):
//   x buffer: 32KB (ELEMENTS_PER_TILE elements)
//   even buffer: 16KB (ELEMENTS_PER_TILE/2 elements)
//   odd buffer: 16KB (ELEMENTS_PER_TILE/2 elements)
// Total per set: ~64KB + alignment gaps. Two sets: ~130KB (fits in 192KB UB).
constexpr unsigned X_PING = 0x00000;
constexpr unsigned EVEN_PING = (X_PING + 0x8000 + 0x100);
constexpr unsigned ODD_PING = (EVEN_PING + 0x4000 + 0x100);
constexpr unsigned X_PONG = (ODD_PING + 0x4000 + 0x100);
constexpr unsigned EVEN_PONG = (X_PONG + 0x8000 + 0x100);
constexpr unsigned ODD_PONG = (EVEN_PONG + 0x4000 + 0x100);

// Fast Hadamard
template <typename T>
AICORE void runTFastHadamard(__gm__ T *x, uint32_t batch, uint32_t n,
                             uint32_t log2_n) {
#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (n == 0 || n > ELEMENTS_PER_TILE) {
    return;
  }

  const uint32_t num_cores = block_num;
  const uint32_t samples_per_core = DIV_ROUNDUP(batch, num_cores);
  const uint32_t sample_offset = samples_per_core * block_idx;
  if (sample_offset >= batch) {
    return;
  }

  uint32_t samples_to_process = samples_per_core;
  if (sample_offset + samples_to_process > batch) {
    samples_to_process = batch - sample_offset;
  }
  if (samples_to_process == 0) {
    return;
  }

  using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;

  using FullTile =
      Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;

  using HalfTile = Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE / 2,
                        BLayout::RowMajor, -1, -1>;

  const uint32_t samples_per_load =
      (n < ELEMENTS_PER_TILE) ? ELEMENTS_PER_TILE / n : 1;

  const uint32_t n_half = n >> 1;

  // Initialize event flags for double buffering
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  uint32_t gm_offset = sample_offset * n;

  for (uint32_t sample_done = 0, ping = 1; sample_done < samples_to_process;
       sample_done += samples_per_load) {
    const event_t ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;

    // Fix tail batch
    uint32_t cur_samples = samples_per_load;
    if (sample_done + cur_samples > samples_to_process) {
      cur_samples = samples_to_process - sample_done;
    }
    uint32_t elements_to_load = cur_samples * n;

    // Select ping or pong UB base addresses
    const unsigned x_base = ping ? X_PING : X_PONG;
    const unsigned even_base = ping ? EVEN_PING : EVEN_PONG;
    const unsigned odd_base = ping ? ODD_PING : ODD_PONG;

    FullTile xBulkTile(1, elements_to_load);
    TASSIGN(xBulkTile, x_base);

    GlobalData xGlobal(x + gm_offset);
    TASSIGN(xGlobal, (x + gm_offset));

    // Shared even/odd scratch tiles (reused across rows and iterations)
    HalfTile evenTile(1, n_half);
    HalfTile oddTile(1, n_half);
    TASSIGN(evenTile, even_base);
    TASSIGN(oddTile, odd_base);

    // Wait for this buffer set to be available (previous store completed)
    wait_flag(PIPE_V, PIPE_MTE2, ev);

    // DMA: GM -> UB (MTE2 pipeline)
    TLOAD(xBulkTile, xGlobal);

    set_flag(PIPE_MTE2, PIPE_V, ev);
    wait_flag(PIPE_MTE2, PIPE_V, ev);

    // Wait for any previous store on this buffer to complete
    wait_flag(PIPE_MTE3, PIPE_V, ev);

    // Process each row with vectorized FHT
    for (uint32_t s = 0; s < cur_samples; ++s) {
      unsigned row_base = x_base + s * n * sizeof(T);

      // Full-row tile for gather source
      FullTile xRowTile(1, n);
      TASSIGN(xRowTile, row_base);

      // Half-row tiles for writing butterfly results
      HalfTile xFirstHalf(1, n_half);
      HalfTile xSecondHalf(1, n_half);
      TASSIGN(xFirstHalf, row_base);
      TASSIGN(xSecondHalf, row_base + n_half * sizeof(T));

      for (uint32_t iter_m = 0; iter_m < log2_n; ++iter_m) {
        // Vectorized even/odd gather
        TGATHER<HalfTile, FullTile, MaskPattern::P0101>(evenTile, xRowTile);
        TGATHER<HalfTile, FullTile, MaskPattern::P1010>(oddTile, xRowTile);

        pipe_barrier(PIPE_V);

        // Vectorized butterfly
        TADD(xFirstHalf, evenTile, oddTile);
        TSUB(xSecondHalf, evenTile, oddTile);

        pipe_barrier(PIPE_V);
      }
    }

    // DMA: UB -> GM (MTE3 pipeline)
    set_flag(PIPE_V, PIPE_MTE3, ev);
    wait_flag(PIPE_V, PIPE_MTE3, ev);

    TSTORE(xGlobal, xBulkTile);

    set_flag(PIPE_MTE3, PIPE_V, ev);
    set_flag(PIPE_V, PIPE_MTE2, ev);

    gm_offset += elements_to_load;
    ping = 1 - ping;
  }

  // Drain remaining event flags
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
#else
  // Cube branch: do nothing
#endif
}

__global__ AICORE void fast_hadamard_fp16(__gm__ void *x, uint32_t batch,
                                          uint32_t n, uint32_t log2_n) {
  runTFastHadamard<half>((__gm__ half *)x, batch, n, log2_n);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                            uint32_t batch, uint32_t n, uint32_t log2_n) {
  fast_hadamard_fp16<<<blockDim, nullptr, stream>>>(x, batch, n, log2_n);
}
