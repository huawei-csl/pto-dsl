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

__global__ AICORE void sync_kernel_dyn(__gm__ float* v1, __gm__ float* v2, int32_t v3) {
  unsigned v4 = 0;
  const int32_t v5 = 0;
  const int32_t v6 = 1;
  const int32_t v7 = 32;
  const int64_t v8 = 0;
  const int64_t v9 = 128;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v10 = get_block_idx();
  int64_t v11 = get_subblockid();
  int64_t v12 = get_subblockdim();
  int64_t v13 = (int64_t) v12;
  int64_t v14 = get_block_num();
  int32_t v15 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v14) * (uint64_t) v13);
  int32_t v16 = v3 / v15;
  int32_t v17 = v3 % v15 != v5 && v3 < v5 == v15 < v5 ? v16 + v6 : v16;
  int32_t v18 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v10) * (uint64_t) v13) + (uint64_t) ((int64_t) v11))) * (uint32_t) v17);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v18 < v3) {
    int32_t v19 = (int32_t) ((uint32_t) v18 + (uint32_t) v17);
    int32_t v20 = (uint32_t) v19 < (uint32_t) v3 ? v19 : v3;
    int32_t v21 = (int32_t) ((uint32_t) v20 - (uint32_t) v18);
    int32_t v22 = v21 / v7;
    for (size_t v23 = (size_t) v5; v23 < ((size_t) (v21 % v7 != v5 && v21 < v5 == v7 < v5 ? v22 + v6 : v22)); v23 += (size_t) v6) {
      int32_t v24 = (int32_t) ((uint32_t) v18 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v23) * (uint32_t) v7));
      int32_t v25 = (int32_t) ((uint32_t) v20 - (uint32_t) v24);
      int32_t v26 = (uint32_t) v25 < (uint32_t) v7 ? v25 : v7;
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v27;
      TASSIGN(v27, v8);
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v28 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v6, v26);
      __ubuf__ float* v29 = v27.data();
      uint64_t v30 = reinterpret_cast<uint64_t>(v29);
      TASSIGN(v28, v30);
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v31;
      TASSIGN(v31, v9);
      Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v6, v26);
      __ubuf__ float* v33 = v31.data();
      uint64_t v34 = reinterpret_cast<uint64_t>(v33);
      TASSIGN(v32, v34);
      pto::Shape<1, 1, 1, 1, 32> v35 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v36 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v4 + (unsigned) v24 * (unsigned) v6), v35, v36);
      pto::Shape<1, 1, 1, 1, 32> v38 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v39 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v4 + (unsigned) v24 * (unsigned) v6), v38, v39);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      TLOAD(v28, v37);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      TRELU(v32, v28);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      pipe_barrier(PIPE_MTE3);
      TSTORE(v40, v32);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
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
