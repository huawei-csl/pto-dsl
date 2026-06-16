#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

__global__ AICORE void RunTMATMULSplitK(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, bool v5, int32_t v6) {
  unsigned v7 = 0;
  const int32_t v8 = 0;
  const int32_t v9 = 1;
  const int32_t v10 = 128;
  const int32_t v11 = 32;
  const int32_t v12 = 4;
  const int64_t v13 = 16384;
  const int64_t v14 = 0;
  const int64_t v15 = 32768;
  using T = float;
  size_t v16 = (size_t) v9;

  #if defined(__DAV_CUBE__)
  int32_t v17 = (int32_t) ((uint32_t) v6 * (uint32_t) v10);
  int64_t v18 = get_block_num();
  int32_t v19 = (int32_t) ((int64_t) v18);
  int32_t v20 = v6 / v19;
  int32_t v21 = v6 % v19 != v8 && v6 < v8 == v19 < v8 ? v20 + v9 : v20;
  int64_t v22 = get_block_idx();
  int32_t v23 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v22) * (uint32_t) v21);
  int32_t v24 = (int32_t) ((uint32_t) v23 + (uint32_t) v21);
  Tile<TileType::Mat, float, 128, 32, BLayout::ColMajor, 128, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v25;
  TASSIGN(v25, v13);
  Tile<TileType::Mat, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v26;
  TASSIGN(v26, v14);
  Tile<TileType::Mat, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v27;
  TASSIGN(v27, v15);
  Tile<TileType::Left, float, 128, 32, BLayout::RowMajor, 128, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v28;
  TASSIGN(v28, v14);
  Tile<TileType::Right, float, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v29;
  TASSIGN(v29, v14);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v30;
  TASSIGN(v30, v14);
  Tile<TileType::Bias, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v31;
  TASSIGN(v31, v14);
  for (size_t v32 = (size_t) v23; v32 < ((size_t) ((uint32_t) v24 < (uint32_t) v6 ? v24 : v6)); v32 += v16) {
    int32_t v33 = (int32_t) ((uint32_t) ((int32_t) v32) * (uint32_t) v10);
    for (size_t v34 = (size_t) v8; v34 < ((size_t) v12); v34 += v16) {
      int32_t v35 = (int32_t) v34;
      int32_t v36 = (int32_t) ((uint32_t) v35 * (uint32_t) v11);
      pto::Shape<1, 1, 1, 128, 32> v37 = pto::Shape<1, 1, 1, 128, 32>();
      pto::Stride<16384, 16384, 16384, 128, 1> v38 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 128, 32>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 32>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v33 * (unsigned) v10 + (unsigned) v36 * (unsigned) v9), v37, v38);
      pto::Shape<1, 1, 1, 32, 128> v40 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<4096, 4096, 4096, 128, 1> v41 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v36 * (unsigned) v10 + v7 * (unsigned) v9), v40, v41);
      pto::Shape<1, 1, 1, 1, 128> v43 = pto::Shape<1, 1, 1, 1, 128>();
      pto::Stride<128, 128, 128, 128, 1> v44 = pto::Stride<128, 128, 128, 128, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v4 + (v7 + v7 * (unsigned) v10 + v7 * (unsigned) v9), v43, v44);
      TLOAD(v25, v39);
      TLOAD(v26, v42);
      if (v5) {
        TLOAD(v27, v45);
      };
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v28, v25);
      TMOV(v29, v26);
      if (v5) {
        TMOV(v31, v27);
      };
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v35 == v8) {
        if (v5) {
          TMATMUL_BIAS(v30, v28, v29, v31);
        } else {
          TMATMUL(v30, v28, v29);
        };
      } else {
        TMATMUL_ACC(v30, v30, v28, v29);
      };
      set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 128, 128> v46 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v47 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v33 * (unsigned) v10 + v7 * (unsigned) v9), v46, v47);
    TSTORE(v48, v30);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  #endif // __DAV_CUBE__

  return;
}
