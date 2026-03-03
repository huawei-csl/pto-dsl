#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void float16_n32(__gm__ half* v1, __gm__ half* v2, int32_t v3) {
  unsigned v4 = 16;
  unsigned v5 = 32;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 16;
  int32_t v9 = 32;
  int32_t v10 = 1;
  int32_t v11 = 0;
  int64_t v12 = 0;
  int64_t v13 = 64;
  int64_t v14 = 160;
  int64_t v15 = 32;
  int64_t v16 = 96;
  using T = float;
  size_t v17 = (size_t) v11;
  size_t v18 = (size_t) v10;
  int64_t v19 = get_block_idx();
  int64_t v20 = get_subblockid();
  int64_t v21 = get_subblockdim();
  int64_t v22 = get_block_num();
  int32_t v23 = (int32_t) ((int64_t) v22);
  int32_t v24 = (int32_t) ((uint32_t) v3 * (uint32_t) v9);
  int32_t v25 = v3 / v23;
  int32_t v26 = v3 % v23 != v11 && v3 < v11 == v23 < v11 ? v25 + v10 : v25;
  int32_t v27 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v19) * (uint64_t) ((int64_t) v21)) + (uint64_t) ((int64_t) v20))) * (uint32_t) v26);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v28;
  TASSIGN(v28, v12);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v13);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v14);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v15);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v32;
  TASSIGN(v32, v16);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v27 < v3) {
    for (size_t v33 = v17; v33 < ((size_t) ((int32_t) ((uint32_t) v27 + (uint32_t) v26) > v3 ? (int32_t) ((uint32_t) v3 - (uint32_t) v27) : v26)); v33 += v18) {
      int32_t v34 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) v33) + (uint32_t) v27) * (uint32_t) v9);
      pto::Shape<1, 1, 1, 1, 32> v35 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v36 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v37 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v34 * (unsigned) v10), v35, v36);
      int32_t v38 = (int32_t) ((uint32_t) v34 + (uint32_t) v8);
      pto::Shape<1, 1, 1, 1, 16> v39 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<16, 16, 16, 16, 1> v40 = pto::Stride<16, 16, 16, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v38 * (unsigned) v10), v39, v40);
      pto::Shape<1, 1, 1, 1, 32> v42 = pto::Shape<1, 1, 1, 1, 32>();
      pto::Stride<32, 32, 32, 32, 1> v43 = pto::Stride<32, 32, 32, 32, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v34 * (unsigned) v10), v42, v43);
      pto::Shape<1, 1, 1, 1, 16> v45 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<16, 16, 16, 16, 1> v46 = pto::Stride<16, 16, 16, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v34 * (unsigned) v10), v45, v46);
      pto::Shape<1, 1, 1, 1, 16> v48 = pto::Shape<1, 1, 1, 1, 16>();
      pto::Stride<16, 16, 16, 16, 1> v49 = pto::Stride<16, 16, 16, 16, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v50 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v38 * (unsigned) v10), v48, v49);
      TLOAD(v32, v37);
      TLOAD(v29, v41);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      pipe_barrier(PIPE_V);
      for (size_t v51 = v17; v51 < v18; v51 += v18) {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TGATHER<Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v28, v32);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TADD(v30, v28, v29);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TSUB(v31, v28, v29);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TSTORE(v47, v30);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_MTE3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TSTORE(v50, v31);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TLOAD(v32, v44);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        pipe_barrier(PIPE_V);
        TLOAD(v29, v50);
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

