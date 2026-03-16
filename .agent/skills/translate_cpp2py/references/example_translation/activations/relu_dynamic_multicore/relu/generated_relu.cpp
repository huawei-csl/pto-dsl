#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void sync_kernel_dyn(__gm__ float* v1, __gm__ float* v2, int32_t v3) {
  unsigned v4 = 32;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 0;
  int32_t v8 = 1;
  int32_t v9 = 32;
  int64_t v10 = 0;
  int64_t v11 = 128;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v12 = get_block_num();
  int32_t v13 = (int32_t) ((int64_t) v12);
  int32_t v14 = v3 / v13;
  int32_t v15 = v3 % v13 != v7 && v3 < v7 == v13 < v7 ? v14 + v8 : v14;
  int64_t v16 = get_block_idx();
  int32_t v17 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v16) * (uint32_t) v15);
  int32_t v18 = (int32_t) ((uint32_t) v17 + (uint32_t) v15);
  int32_t v19 = (uint32_t) v18 < (uint32_t) v3 ? v18 : v3;
  int32_t v20 = (int32_t) ((uint32_t) v19 - (uint32_t) v17);
  int32_t v21 = v20 / v9;
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (size_t v22 = (size_t) v7; v22 < ((size_t) (v20 % v9 != v7 && v20 < v7 == v9 < v7 ? v21 + v8 : v21)); v22 += (size_t) v8) {
    int32_t v23 = (int32_t) ((uint32_t) v17 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v22) * (uint32_t) v9));
    int32_t v24 = (int32_t) ((uint32_t) v19 - (uint32_t) v23);
    int32_t v25 = (uint32_t) v24 < (uint32_t) v9 ? v24 : v9;
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v26 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v25);
    TASSIGN(v26, v10);
    Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v27 = Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v25);
    TASSIGN(v27, v11);
    pto::Shape<1, 1, 1, 1, 32> v28 = pto::Shape<1, 1, 1, 1, 32>();
    pto::Stride<32, 32, 32, 32, 1> v29 = pto::Stride<32, 32, 32, 32, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v23 * (unsigned) v8), v28, v29);
    pto::Shape<1, 1, 1, 1, 32> v31 = pto::Shape<1, 1, 1, 1, 32>();
    pto::Stride<32, 32, 32, 32, 1> v32 = pto::Stride<32, 32, 32, 32, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v33 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v23 * (unsigned) v8), v31, v32);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v26, v30);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TRELU(v27, v26);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pipe_barrier(PIPE_MTE3);
    TSTORE(v33, v27);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

