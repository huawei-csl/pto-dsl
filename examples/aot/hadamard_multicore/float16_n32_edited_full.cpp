#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void float16_n32(__gm__ half* v1, __gm__ half* v2, int32_t v3) {
  unsigned v4 = 16;
  unsigned v5 = 32;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 5;
  int32_t v9 = 16;
  int32_t v10 = 32;
  int32_t v11 = 1;
  int32_t v12 = 0;
  int64_t v13 = 32;
  int64_t v14 = 96;
  int64_t v15 = 64;
  int64_t v16 = 0;
  int64_t v17 = 128;
  using T = float;
  size_t v18 = (size_t) v12;
  size_t v19 = (size_t) v11;
  int64_t v20 = get_block_idx();
  int64_t v21 = get_subblockid();
  int64_t v22 = get_subblockdim();
  int64_t v23 = get_block_num();
  int32_t v24 = (int32_t) ((int64_t) v23);
  int32_t v25 = (int32_t) ((uint32_t) v3 * (uint32_t) v10);
  int32_t v26 = v3 / v24;
  int32_t v27 = v3 % v24 != v12 && v3 < v12 == v24 < v12 ? v26 + v11 : v26;
  int32_t v28 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v20) * (uint64_t) ((int64_t) v22)) + (uint64_t) ((int64_t) v21))) * (uint32_t) v27);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v13);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v14);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v15);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v32;
  TASSIGN(v32, v16);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v33;
  TASSIGN(v33, v17);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v28 < v3) {
    for (size_t v34 = v18; v34 < ((size_t) ((int32_t) ((uint32_t) v28 + (uint32_t) v27) > v3 ? (int32_t) ((uint32_t) v3 - (uint32_t) v28) : v27)); v34 += v19) {
      int32_t v35 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) v34) + (uint32_t) v28) * (uint32_t) v10);
      pto::Shape<1, 1, 1, 1, 32> v36 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v37 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v35 * (unsigned) v11), v36, v37);
      int32_t v39 = (int32_t) ((uint32_t) v35 + (uint32_t) v9);
      pto::Shape<1, 1, 1, 1, 16> v40 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<16, 16, 16, 16, 1> v41 = pto::Stride<16, 16, 16, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v42 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v39 * (unsigned) v11), v40, v41);
      pto::Shape<1, 1, 1, 1, 32> v43 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v44 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v45 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v35 * (unsigned) v11), v43, v44);
      pto::Shape<1, 1, 1, 1, 16> v46 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<16, 16, 16, 16, 1> v47 = pto::Stride<16, 16, 16, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v48 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v35 * (unsigned) v11), v46, v47);
      pto::Shape<1, 1, 1, 1, 16> v49 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<16, 16, 16, 16, 1> v50 = pto::Stride<16, 16, 16, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v51 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v39 * (unsigned) v11), v49, v50);
      TLOAD(v33, v38);
      TLOAD(v30, v42);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pipe_barrier(PIPE_V);
      for (size_t v52 = v18; v52 < ((size_t) v8); v52 += v19) {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TGATHER<Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v29, v33);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        pipe_barrier(PIPE_V);
        TGATHER<Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v30, v33);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TADD(v31, v29, v30);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TSUB(v32, v29, v30);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TSTORE(v48, v31);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TSTORE(v51, v32);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TLOAD(v33, v45);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TLOAD(v30, v51);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
      };
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

