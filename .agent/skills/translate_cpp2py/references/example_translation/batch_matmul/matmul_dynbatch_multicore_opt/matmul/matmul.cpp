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
  const int32_t v11 = 16384;
  const int64_t v12 = 65536;
  const int64_t v13 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v14 = get_block_num();
  int32_t v15 = (int32_t) ((int64_t) v14);
  int64_t v16 = get_block_idx();
  int32_t v17 = (int32_t) ((int64_t) v16);
  int32_t v18 = v6 / v15;
  int32_t v19 = v6 % v15;
  int32_t v20 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v17 * (uint32_t) v18) + (uint32_t) ((uint32_t) v17 < (uint32_t) v19 ? v17 : v19));
  int32_t v21 = (int32_t) ((uint32_t) v20 + (uint32_t) ((int32_t) (uint32_t) v18 + (uint32_t) (v17 < v19 ? v9 : v8)));
  Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v22;
  TASSIGN(v22, v12);
  Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v23;
  TASSIGN(v23, v13);
  Tile<TileType::Left, float, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v24;
  TASSIGN(v24, v13);
  Tile<TileType::Right, float, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v25;
  TASSIGN(v25, v13);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v26;
  TASSIGN(v26, v13);
  pto::Shape<1, 1, 1, 128, 128> v27 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v28 = pto::Stride<16384, 16384, 16384, 128, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v29 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v7 + v7 * (unsigned) v10 + v7 * (unsigned) v9), v27, v28);
  TLOAD(v23, v29);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v25, v23);
  for (size_t v30 = (size_t) v20; v30 < ((size_t) ((uint32_t) v21 < (uint32_t) v6 ? v21 : v6)); v30 += (size_t) v9) {
    int32_t v31 = (int32_t) v30;
    pto::Shape<1, 1, 1, 128, 128> v32 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v33 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + ((v7 + (unsigned) v31 * (unsigned) v11) + v7 * (unsigned) v10 + v7 * (unsigned) v9), v32, v33);
    pto::Shape<1, 1, 1, 128, 128> v35 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v36 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + ((v7 + (unsigned) v31 * (unsigned) v11) + v7 * (unsigned) v10 + v7 * (unsigned) v9), v35, v36);
    TLOAD(v22, v34);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v24, v22);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(v26, v24, v25);
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v37, v26);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  #endif // __DAV_CUBE__

  return;
}
