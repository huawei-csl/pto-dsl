#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void float16_dynamic(__gm__ half* v1, __gm__ half* v2, int32_t v3, int32_t v4, int32_t v5) {
  unsigned v6 = 16;
  unsigned v7 = 32;
  unsigned v8 = 1;
  unsigned v9 = 0;
  int32_t v10 = 16;
  int32_t v11 = 32;
  int32_t v12 = 1;
  int32_t v13 = 0;
  int64_t v14 = 0;
  int64_t v15 = 96;
  int64_t v16 = 64;
  int64_t v17 = 32;
  int64_t v18 = 128;
  using T = float;
  size_t v19 = (size_t) v13;
  size_t v20 = (size_t) v12;
  int64_t v21 = get_block_idx();
  int64_t v22 = get_subblockid();
  int64_t v23 = get_subblockdim();
  int64_t v24 = get_block_num();
  int32_t v25 = (int32_t) ((int64_t) v24);
  int32_t v26 = (int32_t) ((uint32_t) v3 * (uint32_t) v4);
  int32_t v27 = v4 / v11;
  int32_t v28 = v3 / v25;
  int32_t v29 = v3 % v25 != v13 && v3 < v13 == v25 < v13 ? v28 + v12 : v28;
  int32_t v30 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v21) * (uint64_t) ((int64_t) v23)) + (uint64_t) ((int64_t) v22))) * (uint32_t) v29);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v14);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v32;
  TASSIGN(v32, v15);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v33;
  TASSIGN(v33, v16);
  Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v34;
  TASSIGN(v34, v17);
  Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v35;
  TASSIGN(v35, v18);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v30 < v3) {
    for (size_t v36 = v19; v36 < ((size_t) ((int32_t) ((uint32_t) v30 + (uint32_t) v29) > v3 ? (int32_t) ((uint32_t) v3 - (uint32_t) v30) : v29)); v36 += v20) {
      for (size_t v37 = v19; v37 < ((size_t) (v4 % v11 != v13 && v4 < v13 == v11 < v13 ? v27 + v12 : v27)); v37 += v20) {
        int32_t v38 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) v36) + (uint32_t) v30) * (uint32_t) v4) + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v37) * (uint32_t) v11));
        pto::Shape<1, 1, 1, 1, 32> v39 = pto::Shape<1, 1, 1, 1, 32>();
        pto::Stride<32, 32, 32, 32, 1> v40 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v9 + (unsigned) v38 * (unsigned) v12), v39, v40);
        pto::Shape<1, 1, 1, 1, 32> v42 = pto::Shape<1, 1, 1, 1, 32>();
        pto::Stride<32, 32, 32, 32, 1> v43 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v9 + (unsigned) v38 * (unsigned) v12), v42, v43);
        pto::Shape<1, 1, 1, 1, 16> v45 = pto::Shape<1, 1, 1, 1, 16>();
        pto::Stride<16, 16, 16, 16, 1> v46 = pto::Stride<16, 16, 16, 16, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v9 + (unsigned) v38 * (unsigned) v12), v45, v46);
        pto::Shape<1, 1, 1, 1, 16> v48 = pto::Shape<1, 1, 1, 1, 16>();
        pto::Stride<16, 16, 16, 16, 1> v49 = pto::Stride<16, 16, 16, 16, 1>();
        GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND> v50 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>, pto::Layout::ND>(v2 + (v9 + (unsigned) ((int32_t) (uint32_t) v38 + (uint32_t) v10) * (unsigned) v12), v48, v49);
        TLOAD(v35, v41);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        for (size_t v51 = v19; v51 < ((size_t) v5); v51 += v20) {
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          TGATHER<Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v31, v35);
          TGATHER<Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v32, v35);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          TADD(v33, v31, v32);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
          TSUB(v34, v31, v32);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          TSTORE(v47, v33);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          pipe_barrier(PIPE_MTE3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          TSTORE(v50, v34);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TLOAD(v35, v44);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        };
      };
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

