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

__global__ AICORE void tri_inv_trick_fp16(__gm__ float* v1, __gm__ half* v2, __gm__ half* v3, int32_t v4, int32_t v5) {
  unsigned v6 = 0;
  const int32_t v7 = 0;
  const int32_t v8 = 1;
  const int32_t v9 = 64;
  const int64_t v10 = 0;
  const int64_t v11 = 16384;
  const int64_t v12 = 24576;
  const int64_t v13 = 8192;
  using T = float;
  size_t v14 = (size_t) v8;

  #if defined(__DAV_CUBE__)
  int64_t v15 = get_block_idx();
  int32_t v16 = (int32_t) ((int64_t) v15);
  int64_t v17 = get_block_num();
  int32_t v18 = (int32_t) ((int64_t) v17);
  int32_t v19 = (int32_t) ((uint32_t) v4 * (uint32_t) v9);
  int32_t v20 = v4 / v18;
  int32_t v21 = v4 % v18;
  int32_t v22 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v16 * (uint32_t) v20) + (uint32_t) ((uint32_t) v16 < (uint32_t) v21 ? v16 : v21));
  int32_t v23 = (int32_t) ((uint32_t) v22 + (uint32_t) ((int32_t) (uint32_t) v20 + (uint32_t) (v16 < v21 ? v8 : v7)));
  pto::Shape<1, 1, 1, 64, 64> v24 = pto::Shape<1, 1, 1, 64, 64>();
  pto::Stride<4096, 4096, 4096, 64, 1> v25 = pto::Stride<4096, 4096, 4096, 64, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND> v26 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND>(v3 + (v6 + v6 * (unsigned) v9 + v6 * (unsigned) v8), v24, v25);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v27;
  TASSIGN(v27, v10);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v28;
  TASSIGN(v28, v11);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v29;
  TASSIGN(v29, v12);
  Tile<TileType::Mat, half, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v30;
  TASSIGN(v30, v13);
  Tile<TileType::Left, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v31;
  TASSIGN(v31, v10);
  Tile<TileType::Right, half, 64, 64, BLayout::RowMajor, 64, 64, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v32;
  TASSIGN(v32, v10);
  Tile<TileType::Acc, float, 64, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v33;
  TASSIGN(v33, v10);
  set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
  TLOAD(v27, v26);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v31, v27);
  TMOV(v32, v27);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v33, v31, v32);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TMOV(v30, v33);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (size_t v34 = (size_t) v22; v34 < ((size_t) ((uint32_t) v23 < (uint32_t) v4 ? v23 : v4)); v34 += v14) {
    int32_t v35 = (int32_t) ((uint32_t) ((int32_t) v34) * (uint32_t) v9);
    pto::Shape<1, 1, 1, 64, 64> v36 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<4096, 4096, 4096, 64, 1> v37 = pto::Stride<4096, 4096, 4096, 64, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v35 * (unsigned) v9 + v6 * (unsigned) v8), v36, v37);
    pto::Shape<1, 1, 1, 64, 64> v39 = pto::Shape<1, 1, 1, 64, 64>();
    pto::Stride<4096, 4096, 4096, 64, 1> v40 = pto::Stride<4096, 4096, 4096, 64, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND> v41 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 64>, pto::Stride<4096, 4096, 4096, 64, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v35 * (unsigned) v9 + v6 * (unsigned) v8), v39, v40);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v29, v38);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    TMOV(v31, v29);
    TMOV(v32, v29);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    TMATMUL(v33, v31, v32);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    TMOV(v29, v33);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    TMOV(v32, v27);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    TMATMUL(v33, v31, v32);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    TMOV(v31, v27);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    TMATMUL_ACC(v33, v33, v31, v32);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    TMOV(v28, v33);
    set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
    for (size_t v42 = (size_t) v7; v42 < ((size_t) v5); v42 += v14) {
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
      TMOV(v31, v28);
      TMOV(v32, v30);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
      TMATMUL(v33, v31, v32);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      TMOV(v32, v29);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      TMATMUL_ACC(v33, v33, v31, v32);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      if ((int32_t) ((uint32_t) ((int32_t) v42) + (uint32_t) v8) < v5) {
        TMOV(v28, v33);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
        TMOV(v31, v29);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
        TMATMUL(v33, v31, v32);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
        TMOV(v29, v33);
      };
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
    TSTORE(v41, v33);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  }
  wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
