#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _kernel(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, int32_t v4, int32_t v5) {
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 16384;
  int32_t v9 = 1;
  int32_t v10 = 0;
  int64_t v11 = 0;
  int64_t v12 = 32768;
  int64_t v13 = 65536;
  int64_t v14 = 98304;
  int64_t v15 = 131072;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v5 > v10) {
    if (v5 <= v8) {
      int64_t v16 = get_block_idx();
      int64_t v17 = get_subblockid();
      int64_t v18 = get_subblockdim();
      int64_t v19 = (int64_t) v18;
      int64_t v20 = get_block_num();
      int32_t v21 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v20) * (uint64_t) v19);
      int32_t v22 = v4 / v21;
      int32_t v23 = v4 % v21 != v10 && v4 < v10 == v21 < v10 ? v22 + v9 : v22;
      int32_t v24 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v16) * (uint64_t) v19) + (uint64_t) ((int64_t) v17))) * (uint32_t) v23);
      int32_t v25 = (int32_t) ((uint32_t) v24 + (uint32_t) v23);
      int32_t v26 = (int32_t) ((uint32_t) ((uint32_t) v25 < (uint32_t) v4 ? v25 : v4) - (uint32_t) v24);
      int32_t v27 = (int32_t) ((uint32_t) v4 * (uint32_t) v5);
      if (v26 > v10) {
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v28 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v5);
        TASSIGN(v28, v11);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v29 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v5);
        TASSIGN(v29, v12);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v30 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v5);
        TASSIGN(v30, v13);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v31 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v5);
        TASSIGN(v31, v14);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v32 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v5);
        TASSIGN(v32, v15);
        for (size_t v33 = (size_t) v10; v33 < ((size_t) v26); v33 += (size_t) v9) {
          int32_t v34 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v24 + (uint32_t) ((int32_t) v33)) * (uint32_t) v5);
          unsigned v35 = (unsigned) v5 * v6;
          pto::Shape<1, 1, 1, 1, -1> v36 = pto::Shape<1, 1, 1, 1, -1>(v5);
          pto::Stride<-1, -1, -1, -1, 1> v37 = pto::Stride<-1, -1, -1, -1, 1>(v35, v35, v35, v35);
          GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v34 * (unsigned) v9), v36, v37);
          unsigned v39 = (unsigned) v5 * v6;
          pto::Shape<1, 1, 1, 1, -1> v40 = pto::Shape<1, 1, 1, 1, -1>(v5);
          pto::Stride<-1, -1, -1, -1, 1> v41 = pto::Stride<-1, -1, -1, -1, 1>(v39, v39, v39, v39);
          GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v42 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v34 * (unsigned) v9), v40, v41);
          unsigned v43 = (unsigned) v5 * v6;
          pto::Shape<1, 1, 1, 1, -1> v44 = pto::Shape<1, 1, 1, 1, -1>(v5);
          pto::Stride<-1, -1, -1, -1, 1> v45 = pto::Stride<-1, -1, -1, -1, 1>(v43, v43, v43, v43);
          GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v46 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v34 * (unsigned) v9), v44, v45);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TLOAD(v28, v38);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          TLOAD(v29, v42);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TSUB(v32, v28, v28);
          pipe_barrier(PIPE_V);
          TEXP(v30, v32);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          TADD(v31, v28, v28);
          pipe_barrier(PIPE_V);
          TEXP(v31, v31);
          pipe_barrier(PIPE_V);
          TSUB(v32, v31, v30);
          pipe_barrier(PIPE_V);
          TADD(v31, v31, v30);
          pipe_barrier(PIPE_V);
          TDIV(v32, v32, v31);
          pipe_barrier(PIPE_V);
          TADD(v31, v30, v32);
          pipe_barrier(PIPE_V);
          TMUL(v31, v28, v31);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TADD(v32, v30, v30);
          pipe_barrier(PIPE_V);
          TDIV(v31, v31, v32);
          pipe_barrier(PIPE_V);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          TMUL(v31, v31, v29);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v46, v31);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        };
      };
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

