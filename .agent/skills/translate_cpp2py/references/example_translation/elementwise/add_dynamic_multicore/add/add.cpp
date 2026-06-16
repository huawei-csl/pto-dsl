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

__global__ AICORE void vec_add_1d_dynamic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 0;
  const int32_t v6 = 8192;
  const int32_t v7 = 1;
  const int32_t v8 = 0;
  const int64_t v9 = 65536;
  const int64_t v10 = 0;
  const int64_t v11 = 32768;
  using T = float;
  int64_t v12 = get_block_idx();
  int64_t v13 = get_subblockid();
  int64_t v14 = get_subblockdim();
  int64_t v15 = (int64_t) v14;
  int64_t v16 = get_block_num();
  int32_t v17 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v16) * (uint64_t) v15);
  int32_t v18 = v4 / v6;
  int32_t v19 = v4 % v6 != v8 && v4 < v8 == v6 < v8 ? v18 + v7 : v18;
  int32_t v20 = v19 / v17;
  int32_t v21 = v19 % v17 != v8 && v19 < v8 == v17 < v8 ? v20 + v7 : v20;
  int32_t v22 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v12) * (uint64_t) v15) + (uint64_t) ((int64_t) v13))) * (uint32_t) v21);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v23;
  TASSIGN(v23, v9);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v24;
  TASSIGN(v24, v10);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v25;
  TASSIGN(v25, v11);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v22 < v19) {
    int32_t v26 = (int32_t) ((uint32_t) v22 + (uint32_t) v21) > v19 ? (int32_t) ((uint32_t) v19 - (uint32_t) v22) : v21;
    if ((int32_t) ((uint32_t) v26 * (uint32_t) v6) > v8) {
      for (size_t v27 = (size_t) v8; v27 < ((size_t) v26); v27 += (size_t) v7) {
        int32_t v28 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) v27) + (uint32_t) v22) * (uint32_t) v6);
        pto::Shape<1, 1, 1, 1, 8192> v29 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v30 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v31 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) v28 * (unsigned) v7), v29, v30);
        pto::Shape<1, 1, 1, 1, 8192> v32 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v33 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v34 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) v28 * (unsigned) v7), v32, v33);
        pto::Shape<1, 1, 1, 1, 8192> v35 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v36 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v3 + (v5 + (unsigned) v28 * (unsigned) v7), v35, v36);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v23, v31);
        TLOAD(v24, v34);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        TADD(v25, v23, v24);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v37, v25);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      };
    };
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
