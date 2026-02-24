#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void vec_gather_2d_dynamic_int32_P1111(__gm__ int32_t* v1, __gm__ int32_t* v2, __gm__ int32_t* v3, int32_t v4) {
  unsigned v5 = 32;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 32;
  int32_t v9 = 1;
  int32_t v10 = 0;
  int64_t v11 = 384;
  int64_t v12 = 256;
  int64_t v13 = 0;
  int64_t v14 = 128;
  using T = float;
  size_t v15 = (size_t) v10;
  size_t v16 = (size_t) v9;
  int64_t v17 = get_block_idx();
  int64_t v18 = get_subblockid();
  int64_t v19 = get_subblockdim();
  int64_t v20 = get_block_num();
  int32_t v21 = (int32_t) ((int64_t) v20);
  int32_t v22 = v4 / v8;
  int32_t v23 = v4 % v8 != v10 && v4 < v10 == v8 < v10 ? v22 + v9 : v22;
  int32_t v24 = v23 / v21;
  int32_t v25 = v23 % v21 != v10 && v23 < v10 == v21 < v10 ? v24 + v9 : v24;
  int32_t v26 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v17) * (uint64_t) ((int64_t) v19)) + (uint64_t) ((int64_t) v18))) * (uint32_t) v25);

  #if defined(__DAV_VEC__)
  Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v11);
  Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v12);
  Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v13);
  Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v14);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v26 < v23) {
    int32_t v31 = (int32_t) ((uint32_t) v23 - (uint32_t) v26);
    if ((int32_t) ((uint32_t) v26 + (uint32_t) v25) > v23) {
      if ((int32_t) ((uint32_t) v31 * (uint32_t) v8) > v10) {
        for (size_t v32 = v15; v32 < ((size_t) v31); v32 += v16) {
          int32_t v33 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) v32) + (uint32_t) v26) * (uint32_t) v8);
          pto::Shape<1, 1, 1, 1, 32> v34 = pto::Shape<1, 1, 1, 1, 32>();
          pto::Stride<32, 32, 32, 32, 1> v35 = pto::Stride<32, 32, 32, 32, 1>();
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v36 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v33 * (unsigned) v9), v34, v35);
          pto::Shape<1, 1, 1, 1, 32> v37 = pto::Shape<1, 1, 1, 1, 32>();
          pto::Stride<32, 32, 32, 32, 1> v38 = pto::Stride<32, 32, 32, 32, 1>();
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v39 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v33 * (unsigned) v9), v37, v38);
          pto::Shape<1, 1, 1, 1, 32> v40 = pto::Shape<1, 1, 1, 1, 32>();
          pto::Stride<32, 32, 32, 32, 1> v41 = pto::Stride<32, 32, 32, 32, 1>();
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v42 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v33 * (unsigned) v9), v40, v41);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TLOAD(v27, v36);
          TLOAD(v28, v39);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TGATHER(v29, v27, v28);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          TGATHER<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1111>(v30, v29);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          TSTORE(v42, v30);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        };
      };
    } else {
      if ((int32_t) ((uint32_t) v25 * (uint32_t) v8) > v10) {
        for (size_t v43 = v15; v43 < ((size_t) v25); v43 += v16) {
          int32_t v44 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) v43) + (uint32_t) v26) * (uint32_t) v8);
          pto::Shape<1, 1, 1, 1, 32> v45 = pto::Shape<1, 1, 1, 1, 32>();
          pto::Stride<32, 32, 32, 32, 1> v46 = pto::Stride<32, 32, 32, 32, 1>();
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v47 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v44 * (unsigned) v9), v45, v46);
          pto::Shape<1, 1, 1, 1, 32> v48 = pto::Shape<1, 1, 1, 1, 32>();
          pto::Stride<32, 32, 32, 32, 1> v49 = pto::Stride<32, 32, 32, 32, 1>();
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v50 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v44 * (unsigned) v9), v48, v49);
          pto::Shape<1, 1, 1, 1, 32> v51 = pto::Shape<1, 1, 1, 1, 32>();
          pto::Stride<32, 32, 32, 32, 1> v52 = pto::Stride<32, 32, 32, 32, 1>();
          GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v53 = GlobalTensor<int32_t, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v44 * (unsigned) v9), v51, v52);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          TLOAD(v27, v47);
          TLOAD(v28, v50);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          TGATHER(v29, v27, v28);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
          TGATHER<Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, int32_t, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1111>(v30, v29);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
          TSTORE(v53, v30);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        };
      };
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

