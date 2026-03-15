#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void vec_add_1d_dynamic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4) {
  unsigned v5 = 8192;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 8192;
  int32_t v9 = 2;
  int32_t v10 = 1;
  int32_t v11 = 0;
  int64_t v12 = 0;
  int64_t v13 = 163840;
  int64_t v14 = 32768;
  int64_t v15 = 98304;
  int64_t v16 = 65536;
  int64_t v17 = 131072;
  using T = float;
  int64_t v18 = get_block_idx();
  int64_t v19 = get_subblockid();
  int64_t v20 = get_subblockdim();
  int64_t v21 = (int64_t) v20;
  int64_t v22 = get_block_num();
  int32_t v23 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v22) * (uint64_t) v21);
  int32_t v24 = v4 / v8;
  int32_t v25 = v4 % v8 != v11 && v4 < v11 == v8 < v11 ? v24 + v10 : v24;
  int32_t v26 = v25 / v23;
  int32_t v27 = v25 % v23 != v11 && v25 < v11 == v23 < v11 ? v26 + v10 : v26;
  int32_t v28 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v18) * (uint64_t) v21) + (uint64_t) ((int64_t) v19))) * (uint32_t) v27);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v12);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v13);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v14);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null> v32;
  TASSIGN(v32, v15);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null> v33;
  TASSIGN(v33, v16);
  Tile<TileType::Vec, float, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null> v34;
  TASSIGN(v34, v17);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  if (v28 < v25) {
    int32_t v35 = (int32_t) ((uint32_t) v28 + (uint32_t) v27) > v25 ? (int32_t) ((uint32_t) v25 - (uint32_t) v28) : v27;
    if ((int32_t) ((uint32_t) v35 * (uint32_t) v8) > v11) {
      for (size_t v36 = (size_t) v11; v36 < ((size_t) v35); v36 += (size_t) v10) {
        int32_t v37 = (int32_t) v36;
        int32_t v38 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v37 + (uint32_t) v28) * (uint32_t) v8);
        pto::Shape<1, 1, 1, 1, 8192> v39 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v40 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v41 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v38 * (unsigned) v10), v39, v40);
        pto::Shape<1, 1, 1, 1, 8192> v42 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v43 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v38 * (unsigned) v10), v42, v43);
        pto::Shape<1, 1, 1, 1, 8192> v45 = pto::Shape<1, 1, 1, 1, 8192>();
        pto::Stride<8192, 8192, 8192, 8192, 1> v46 = pto::Stride<8192, 8192, 8192, 8192, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 8192>, pto::Stride<8192, 8192, 8192, 8192, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v38 * (unsigned) v10), v45, v46);
        if (v37 % v9 == v11) {
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TLOAD(v29, v41);
          TLOAD(v30, v44);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          TADD(v31, v29, v30);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v47, v31);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        } else {
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          TLOAD(v32, v41);
          TLOAD(v33, v44);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
          TADD(v34, v32, v33);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v47, v34);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        };
      };
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
  #endif // __DAV_VEC__

  return;
}

