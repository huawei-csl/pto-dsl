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
  int64_t v12 = 128;
  int64_t v13 = 320;
  int64_t v14 = 256;
  int64_t v15 = 64;
  int64_t v16 = 0;
  using T = float;
  size_t v17 = (size_t) v11;
  size_t v18 = (size_t) v10;
  size_t v19 = (size_t) v9;
  int64_t v20 = get_block_idx();
  int64_t v21 = get_subblockid();
  int64_t v22 = get_subblockdim();
  int64_t v23 = (int64_t) v22;
  int64_t v24 = get_block_num();
  int32_t v25 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v24) * (uint32_t) ((int32_t) v23));
  int32_t v26 = v8 / v25;
  int32_t v27 = v8 % v25 != v11 && v8 < v11 == v25 < v11 ? v26 + v10 : v26;
  int32_t v28 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v20) * (uint64_t) v23) + (uint64_t) ((int64_t) v21))) * (uint32_t) v27);
  size_t v29 = (size_t) v28;
  int32_t v30 = (int32_t) ((uint32_t) v28 + (uint32_t) v27);

  #if defined(__DAV_VEC__)
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v12);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v32;
  TASSIGN(v32, v13);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v33;
  TASSIGN(v33, v14);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v34;
  TASSIGN(v34, v15);
  Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null> v35;
  TASSIGN(v35, v16);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  if (v28 < v8) {
    if (v30 > v8) {
      for (size_t v36 = v29; v36 < ((size_t) v8); v36 += v18) {
        int32_t v37 = (int32_t) v36;
        pto::Shape<1, 1, 1, 1, 32> v38 = pto::Shape<1, 1, 1, 1, 32>();
        pto::Stride<32, 32, 32, 32, 1> v39 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v37 * (unsigned) v8 + v6 * (unsigned) v10), v38, v39);
        pto::Shape<1, 1, 1, 1, 32> v41 = pto::Shape<1, 1, 1, 1, 32>();
        pto::Stride<32, 32, 32, 32, 1> v42 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v43 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v37 * (unsigned) v8 + v6 * (unsigned) v10), v41, v42);
        pto::Shape<1, 1, 1, 1, 16> v44 = pto::Shape<1, 1, 1, 1, 16>();
        pto::Stride<32, 32, 32, 32, 1> v45 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v46 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v37 * (unsigned) v8 + v6 * (unsigned) v10), v44, v45);
        pto::Shape<1, 1, 1, 1, 16> v47 = pto::Shape<1, 1, 1, 1, 16>();
        pto::Stride<32, 32, 32, 32, 1> v48 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v49 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v37 * (unsigned) v8 + v3 * (unsigned) v10), v47, v48);
        TLOAD(v31, v40);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(v43, v31);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        for (size_t v50 = v17; v50 < v19; v50 += v18) {
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TLOAD(v31, v43);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
          TGATHER<Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v32, v31);
          TGATHER<Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v33, v31);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TADD(v34, v32, v33);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          TSUB(v35, v32, v33);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID3);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v46, v34);
          pipe_barrier(PIPE_MTE3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
          TSTORE(v49, v35);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        };
      };
    } else {
      for (size_t v51 = v29; v51 < ((size_t) v30); v51 += v18) {
        int32_t v52 = (int32_t) v51;
        pto::Shape<1, 1, 1, 1, 32> v53 = pto::Shape<1, 1, 1, 1, 32>();
        pto::Stride<32, 32, 32, 32, 1> v54 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v55 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v52 * (unsigned) v8 + v6 * (unsigned) v10), v53, v54);
        pto::Shape<1, 1, 1, 1, 32> v56 = pto::Shape<1, 1, 1, 1, 32>();
        pto::Stride<32, 32, 32, 32, 1> v57 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v58 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v52 * (unsigned) v8 + v6 * (unsigned) v10), v56, v57);
        pto::Shape<1, 1, 1, 1, 16> v59 = pto::Shape<1, 1, 1, 1, 16>();
        pto::Stride<32, 32, 32, 32, 1> v60 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v61 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v52 * (unsigned) v8 + v6 * (unsigned) v10), v59, v60);
        pto::Shape<1, 1, 1, 1, 16> v62 = pto::Shape<1, 1, 1, 1, 16>();
        pto::Stride<32, 32, 32, 32, 1> v63 = pto::Stride<32, 32, 32, 32, 1>();
        GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v64 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 16>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v52 * (unsigned) v8 + v3 * (unsigned) v10), v62, v63);
        TLOAD(v31, v55);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID4);
        TSTORE(v58, v31);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
        for (size_t v65 = v17; v65 < v19; v65 += v18) {
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
          TLOAD(v31, v58);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
          set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);
          TGATHER<Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v32, v31);
          TGATHER<Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, 1, 16, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v33, v31);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
          TADD(v34, v32, v33);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
          TSUB(v35, v32, v33);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
          wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID7);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v61, v34);
          pipe_barrier(PIPE_MTE3);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
          TSTORE(v64, v35);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        };
      };
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  #endif // __DAV_VEC__

  return;
}

