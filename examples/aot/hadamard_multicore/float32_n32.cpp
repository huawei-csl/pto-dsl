#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void float32_n32(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 16;
  unsigned v4 = 32;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 5;
  int32_t v8 = 16;
  int32_t v9 = 32;
  int32_t v10 = 1;
  int32_t v11 = 0;
  int64_t v12 = 64;
  int64_t v13 = 0;
  int64_t v14 = 128;
  int64_t v15 = 320;
  int64_t v16 = 192;
  using T = float;
  size_t v17 = (size_t) v10;
  int64_t v18 = get_block_idx();
  int64_t v19 = get_subblockid();
  int64_t v20 = get_subblockdim();
  int64_t v21 = get_block_num();
  int32_t v22 = (int32_t) ((int64_t) v21);
  int32_t v23 = v9 / v22;
  int32_t v24 = v9 % v22 != v11 && v9 < v11 == v22 < v11 ? v23 + v10 : v23;
  int32_t v25 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v18) * (uint64_t) ((int64_t) v20)) + (uint64_t) ((int64_t) v19))) * (uint32_t) v24);
  int32_t v26 = (int32_t) ((uint32_t) v25 + (uint32_t) v24);

  #if defined(__DAV_VEC__)
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v27;
  TASSIGN(v27, v12);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v13);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v14);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v15);
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v16);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  if (v25 < v9) {
    for (size_t v32 = (size_t) v25; v32 < ((size_t) (v26 > v9 ? v9 : v26)); v32 += v17) {
      int32_t v33 = (int32_t) v32;
      pto::Shape<1, 1, 1, 1, 32> v34 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v35 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v36 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v33 * (unsigned) v9 + v6 * (unsigned) v10), v34, v35);
      pto::Shape<1, 1, 1, 1, 32> v37 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v38 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v39 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v33 * (unsigned) v9 + v6 * (unsigned) v10), v37, v38);
      pto::Shape<1, 1, 1, 1, 16> v40 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<32, 32, 32, 32, 1> v41 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v33 * (unsigned) v9 + v6 * (unsigned) v10), v40, v41);
      pto::Shape<1, 1, 1, 1, 16> v43 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<32, 32, 32, 32, 1> v44 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v33 * (unsigned) v9 + v3 * (unsigned) v10), v43, v44);
      for (size_t v46 = (size_t) v11; v46 < ((size_t) v7); v46 += v17) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        if ((int32_t) v46 == v11) {
          TLOAD(v31, v36);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
        } else {
          TLOAD(v31, v39);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        };
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TGATHER<Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v27, v31);
        TGATHER<Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v28, v31);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        TADD(v29, v27, v28);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSUB(v30, v27, v28);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID1);
        TSTORE(v42, v29);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        TSTORE(v45, v30);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      };
    };
  } else {
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

