#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _kernel(__gm__ half* v1, __gm__ half* v2, int32_t v3, int32_t v4) {
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 16384;
  int32_t v8 = 1;
  int32_t v9 = 0;
  int64_t v10 = 0;
  int64_t v11 = 32768;
  int64_t v12 = 65536;
  int64_t v13 = 98304;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v4 > v9) {
    if (v4 <= v7) {
      int64_t v14 = get_block_idx();
      int64_t v15 = get_subblockid();
      int64_t v16 = get_subblockdim();
      int64_t v17 = (int64_t) v16;
      int64_t v18 = get_block_num();
      int32_t v19 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v18) * (uint64_t) v17);
      int32_t v20 = v3 / v19;
      int32_t v21 = v3 % v19 != v9 && v3 < v9 == v19 < v9 ? v20 + v8 : v20;
      int32_t v22 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v14) * (uint64_t) v17) + (uint64_t) ((int64_t) v15))) * (uint32_t) v21);
      int32_t v23 = (int32_t) ((uint32_t) v22 + (uint32_t) v21);
      int32_t v24 = (int32_t) ((uint32_t) ((uint32_t) v23 < (uint32_t) v3 ? v23 : v3) - (uint32_t) v22);
      int32_t v25 = (int32_t) ((uint32_t) v3 * (uint32_t) v4);
      if (v24 > v9) {
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v26 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4);
        TASSIGN(v26, v10);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v27 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4);
        TASSIGN(v27, v11);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v28 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4);
        TASSIGN(v28, v12);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v29 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4);
        TASSIGN(v29, v13);
        for (size_t v30 = (size_t) v9; v30 < ((size_t) v24); v30 += (size_t) v8) {
          int32_t v31 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v22 + (uint32_t) ((int32_t) v30)) * (uint32_t) v4);
          unsigned v32 = (unsigned) v4 * v5;
          pto::Shape<1, 1, 1, 1, -1> v33 = pto::Shape<1, 1, 1, 1, -1>(v4);
          pto::Stride<-1, -1, -1, -1, 1> v34 = pto::Stride<-1, -1, -1, -1, 1>(v32, v32, v32, v32);
          GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v31 * (unsigned) v8), v33, v34);
          unsigned v36 = (unsigned) v4 * v5;
          pto::Shape<1, 1, 1, 1, -1> v37 = pto::Shape<1, 1, 1, 1, -1>(v4);
          pto::Stride<-1, -1, -1, -1, 1> v38 = pto::Stride<-1, -1, -1, -1, 1>(v36, v36, v36, v36);
          GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v39 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v31 * (unsigned) v8), v37, v38);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TLOAD(v26, v35);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TSUB(v29, v26, v26);
          pipe_barrier(PIPE_V);
          TEXP(v27, v29);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          TSUB(v28, v29, v26);
          pipe_barrier(PIPE_V);
          TEXP(v28, v28);
          pipe_barrier(PIPE_V);
          TADD(v28, v28, v27);
          pipe_barrier(PIPE_V);
          TDIV(v28, v27, v28);
          pipe_barrier(PIPE_V);
          TMUL(v28, v26, v28);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v39, v28);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        };
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

