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

__global__ AICORE void vec_add_kernel_2d_dynamic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4, int32_t v5) {
  unsigned v6 = 0;
  const int32_t v7 = 32;
  const int32_t v8 = 1;
  const int64_t v9 = 0;
  const int64_t v10 = 4096;
  const int64_t v11 = 8192;
  using T = float;
  int64_t v12 = get_block_idx();
  int64_t v13 = get_subblockid();
  int64_t v14 = get_subblockdim();
  int32_t v15 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v12) * (uint64_t) ((int64_t) v14)) + (uint64_t) ((int64_t) v13))) * (uint32_t) v7);
  pto::Shape<1, 1, 1, 32, 32> v16 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v17 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v18 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v15 * (unsigned) v7 + v6 * (unsigned) v8), v16, v17);
  pto::Shape<1, 1, 1, 32, 32> v19 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v20 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v21 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v15 * (unsigned) v7 + v6 * (unsigned) v8), v19, v20);
  pto::Shape<1, 1, 1, 32, 32> v22 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v23 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v3 + (v6 + (unsigned) v15 * (unsigned) v7 + v6 * (unsigned) v8), v22, v23);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v25;
  TASSIGN(v25, v9);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v26 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v4, v5);
  __ubuf__ float* v27 = v25.data();
  uint64_t v28 = reinterpret_cast<uint64_t>(v27);
  TASSIGN(v26, v28);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v29;
  TASSIGN(v29, v10);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v30 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v4, v5);
  __ubuf__ float* v31 = v29.data();
  uint64_t v32 = reinterpret_cast<uint64_t>(v31);
  TASSIGN(v30, v32);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33;
  TASSIGN(v33, v11);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v34 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v4, v5);
  __ubuf__ float* v35 = v33.data();
  uint64_t v36 = reinterpret_cast<uint64_t>(v35);
  TASSIGN(v34, v36);
  TLOAD(v26, v18);
  TLOAD(v30, v21);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(v34, v26, v30);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v24, v34);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
