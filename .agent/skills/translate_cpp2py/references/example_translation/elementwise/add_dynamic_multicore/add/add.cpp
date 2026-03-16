#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void vec_add_1d_dynamic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 8192;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 8192;
  int32_t v9 = 1;
  int32_t v10 = 0;
  int64_t v11 = 65536;
  int64_t v12 = 32768;
  int64_t v13 = 0;
  using T = float;
  int64_t v14 = get_block_idx();
  int64_t v15 = get_subblockid();
  int64_t v16 = get_subblockdim();
  int64_t v17 = (int64_t) v16;
  int64_t v18 = get_block_num();
  int32_t v19 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v18) * (uint64_t) v17);
  int32_t v20 = v4 / v8;
  int32_t v21 = v4 % v8 != v10 && v4 < v10 == v8 < v10 ? v20 + v9 : v20;
  int32_t v22 = v21 / v19;
  int32_t v23 = v21 % v19 != v10 && v21 < v10 == v19 < v10 ? v22 + v9 : v22;
  int32_t v24 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v14) * (uint64_t) v17) + (uint64_t) ((int64_t) v15))) * (uint32_t) v23);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v11);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v12);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v13);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v24 < v21) {
    int32_t v28 = (int32_t) ((uint32_t) v24 + (uint32_t) v23) > v21 ? (int32_t) ((uint32_t) v21 - (uint32_t) v24) : v23;
    if ((int32_t) ((uint32_t) v28 * (uint32_t) v8) > v10) {
      for (size_t v29 = (size_t) v10; v29 < ((size_t) v28); v29 += (size_t) v9) {
        int32_t v30 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) v29) + (uint32_t) v24) * (uint32_t) v8);
        pto::Shape<1, 1, 1, 1, 8192> v31 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v32 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v33 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v30 * (unsigned) v9), v31, v32);
        pto::Shape<1, 1, 1, 1, 8192> v34 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v35 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v36 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v30 * (unsigned) v9), v34, v35);
        pto::Shape<1, 1, 1, 1, 8192> v37 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v38 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v30 * (unsigned) v9), v37, v38);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v25, v33);
        TLOAD(v26, v36);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        TADD(v27, v25, v26);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v39, v27);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      };
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

