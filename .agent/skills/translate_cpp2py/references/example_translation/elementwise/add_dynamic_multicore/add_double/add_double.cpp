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
  const int32_t v7 = 2;
  const int32_t v8 = 1;
  const int32_t v9 = 0;
  const int64_t v10 = 98304;
  const int64_t v11 = 32768;
  const int64_t v12 = 131072;
  const int64_t v13 = 0;
  const int64_t v14 = 65536;
  const int64_t v15 = 163840;
  using T = float;
  int64_t v16 = get_block_idx();
  int64_t v17 = get_subblockid();
  int64_t v18 = get_subblockdim();
  int64_t v19 = (int64_t) v18;
  int64_t v20 = get_block_num();
  int32_t v21 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v20) * (uint64_t) v19);
  int32_t v22 = v4 / v6;
  int32_t v23 = v4 % v6 != v9 && v4 < v9 == v6 < v9 ? v22 + v8 : v22;
  int32_t v24 = v23 / v21;
  int32_t v25 = v23 % v21 != v9 && v23 < v9 == v21 < v9 ? v24 + v8 : v24;
  int32_t v26 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v16) * (uint64_t) v19) + (uint64_t) ((int64_t) v17))) * (uint32_t) v25);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v27;
  TASSIGN(v27, v10);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v28;
  TASSIGN(v28, v11);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v29;
  TASSIGN(v29, v12);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30;
  TASSIGN(v30, v13);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v31;
  TASSIGN(v31, v14);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32;
  TASSIGN(v32, v15);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  if (v26 < v23) {
    int32_t v33 = (int32_t) ((uint32_t) v26 + (uint32_t) v25) > v23 ? (int32_t) ((uint32_t) v23 - (uint32_t) v26) : v25;
    if ((int32_t) ((uint32_t) v33 * (uint32_t) v6) > v9) {
      for (size_t v34 = (size_t) v9; v34 < ((size_t) v33); v34 += (size_t) v8) {
        int32_t v35 = (int32_t) v34;
        int32_t v36 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v35 + (uint32_t) v26) * (uint32_t) v6);
        pto::Shape<1, 1, 1, 1, 8192> v37 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v38 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v1 + (v5 + (unsigned) v36 * (unsigned) v8), v37, v38);
        pto::Shape<1, 1, 1, 1, 8192> v40 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v41 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v2 + (v5 + (unsigned) v36 * (unsigned) v8), v40, v41);
        pto::Shape<1, 1, 1, 1, 8192> v43 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v44 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v3 + (v5 + (unsigned) v36 * (unsigned) v8), v43, v44);
        if (v35 % v7 == v9) {
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TLOAD(v27, v39);
          TLOAD(v28, v42);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          TADD(v29, v27, v28);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v45, v29);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        } else {
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          TLOAD(v30, v39);
          TLOAD(v31, v42);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
          TADD(v32, v30, v31);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v45, v32);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        };
      };
    };
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
