#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void float32_n32(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 16;
  unsigned v4 = 32;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 16;
  int32_t v8 = 32;
  int32_t v9 = 5;
  int32_t v10 = 1;
  int32_t v11 = 0;
  int64_t v12 = 64;
  int64_t v13 = 0;
  int64_t v14 = 192;
  int64_t v15 = 256;
  int64_t v16 = 320;
  using T = float;
  size_t v17 = (size_t) v10;
  int64_t v18 = get_block_idx();
  int64_t v19 = get_subblockid();
  int64_t v20 = get_subblockdim();
  int64_t v21 = (int64_t) v20;
  int64_t v22 = get_block_num();
  int32_t v23 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v22) * (uint32_t) ((int32_t) v21));
  int32_t v24 = v8 / v23;
  int32_t v25 = v8 % v23 != v11 && v8 < v11 == v23 < v11 ? v24 + v10 : v24;
  int32_t v26 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v18) * (uint64_t) v21) + (uint64_t) ((int64_t) v19))) * (uint32_t) v25);

  #if defined(__DAV_VEC__)
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v12);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v13);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v14);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v15);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v16);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  if (v26 < v8) {
    int32_t v32 = (int32_t) ((uint32_t) v26 + (uint32_t) v25);
    size_t v33;
    if (v32 > v8) {
      v33 = (size_t) v8;
    } else {
      v33 = (size_t) v32;
    };
    for (size_t v34 = (size_t) v26; v34 < v33; v34 += v17) {
      int32_t v35 = (int32_t) v34;
      pto::Shape<1, 1, 1, 1, 32> v36 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v37 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v38 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v35 * (unsigned) v8 + v6 * (unsigned) v10), v36, v37);
      pto::Shape<1, 1, 1, 1, 32> v39 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v40 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v41 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v35 * (unsigned) v8 + v6 * (unsigned) v10), v39, v40);
      pto::Shape<1, 1, 1, 1, 16> v42 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<32, 32, 32, 32, 1> v43 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v35 * (unsigned) v8 + v6 * (unsigned) v10), v42, v43);
      pto::Shape<1, 1, 1, 1, 16> v45 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<32, 32, 32, 32, 1> v46 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v35 * (unsigned) v8 + v3 * (unsigned) v10), v45, v46);
      TLOAD(v27, v38);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
      TSTORE(v41, v27);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      for (size_t v48 = (size_t) v11; v48 < ((size_t) v9); v48 += v17) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TLOAD(v27, v41);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TGATHER<Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v28, v27);
        TGATHER<Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v29, v27);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TADD(v30, v28, v29);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSUB(v31, v28, v29);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
        pipe_barrier(PIPE_MTE3);
        TSTORE(v44, v30);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        TSTORE(v47, v31);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
      };
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

