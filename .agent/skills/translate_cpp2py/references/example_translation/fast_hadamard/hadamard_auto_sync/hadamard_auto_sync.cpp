#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void fast_hadamard_autosync(__gm__ half* v1, int32_t v2, int32_t v3, int32_t v4) {
  unsigned v5 = 16384;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 2;
  int32_t v9 = 1;
  int32_t v10 = 0;
  int32_t v11 = 8192;
  int64_t v12 = 0;
  int64_t v13 = 32768;
  int64_t v14 = 49152;
  int64_t v15 = 65536;
  int64_t v16 = 98304;
  int64_t v17 = 114688;
  using T = float;
  size_t v18 = (size_t) v10;
  size_t v19 = (size_t) v9;
  size_t v20 = (size_t) v4;
  int64_t v21 = get_block_idx();
  int64_t v22 = get_subblockid();
  int64_t v23 = get_subblockdim();
  int64_t v24 = (int64_t) v23;
  int64_t v25 = get_block_num();
  int32_t v26 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v25) * (uint64_t) v24);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int32_t v27 = v2 / v26;
  int32_t v28 = v2 % v26 != v10 && v2 < v10 == v26 < v10 ? v27 + v9 : v27;
  int32_t v29 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v21) * (uint64_t) v24) + (uint64_t) ((int64_t) v22))) * (uint32_t) v28);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  if (v29 < v2) {
    int32_t v30 = (int32_t) ((uint32_t) v29 + (uint32_t) v28) > v2 ? (int32_t) ((uint32_t) v2 - (uint32_t) v29) : v28;
    if (v30 > v10) {
      int32_t v31 = (int32_t) ((uint32_t) v2 * (uint32_t) v3);
      Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v32 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v3);
      TASSIGN(v32, v12);
      int32_t v33 = v3 / v8;
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v34 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v33);
      TASSIGN(v34, v13);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v35 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v33);
      TASSIGN(v35, v14);
      Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v36 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v3);
      TASSIGN(v36, v15);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v37 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v33);
      TASSIGN(v37, v16);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v38 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v33);
      TASSIGN(v38, v17);
      for (size_t v39 = v18; v39 < ((size_t) v30); v39 += v19) {
        int32_t v40 = (int32_t) v39;
        int32_t v41 = (int32_t) ((uint32_t) v30 - (uint32_t) v40);
        int32_t v42 = v41 < v9 ? v41 : v9;
        size_t v43 = (size_t) v42;
        if (v42 > v10) {
          int32_t v44 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v29 + (uint32_t) v40) * (uint32_t) v3);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          if (v40 % v8 == v10) {
            for (size_t v45 = v18; v45 < v43; v45 += v19) {
              unsigned v46 = (unsigned) v3 * v6;
              pto::Shape<1, 1, 1, 1, -1> v47 = pto::Shape<1, 1, 1, 1, -1>(v3);
              pto::Stride<-1, -1, -1, -1, 1> v48 = pto::Stride<-1, -1, -1, -1, 1>(v46, v46, v46, v46);
              GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v49 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v44 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v45) * (uint32_t) v3)) * (unsigned) v9), v47, v48);
              __ubuf__ half* v50 = v32.data();
              int64_t v51 = (int64_t) v3;
              int32_t v52 = (int32_t) ((int64_t) (uint64_t) v51 - (uint64_t) ((int64_t) (uint64_t) v12 % (uint64_t) v51));
              Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v53 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v52 < v11 ? v52 : v11);
              uint64_t v54 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v50 + (v7 + v7 * v5) + v7 * v6));
              TASSIGN(v53, v54);
              __ubuf__ half* v55 = v32.data();
              int32_t v56 = (int32_t) ((int64_t) (uint64_t) v51 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) % (uint64_t) v51));
              Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v57 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v56 < v11 ? v56 : v11);
              uint64_t v58 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v55 + (v7 + v7 * v5) + (unsigned) v33 * v6));
              TASSIGN(v57, v58);
              wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
              TLOAD(v32, v49);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              for (size_t v59 = v18; v59 < v20; v59 += v19) {
                pipe_barrier(PIPE_V);
                TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v34, v32);
                TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v35, v32);
                pipe_barrier(PIPE_V);
                TADD(v53, v34, v35);
                pipe_barrier(PIPE_V);
                TSUB(v57, v34, v35);
              };
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              pipe_barrier(PIPE_MTE3);
              TSTORE(v49, v32);
              set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            };
          } else {
            for (size_t v60 = v18; v60 < v43; v60 += v19) {
              unsigned v61 = (unsigned) v3 * v6;
              pto::Shape<1, 1, 1, 1, -1> v62 = pto::Shape<1, 1, 1, 1, -1>(v3);
              pto::Stride<-1, -1, -1, -1, 1> v63 = pto::Stride<-1, -1, -1, -1, 1>(v61, v61, v61, v61);
              GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v64 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v44 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v60) * (uint32_t) v3)) * (unsigned) v9), v62, v63);
              __ubuf__ half* v65 = v36.data();
              int64_t v66 = (int64_t) v3;
              int32_t v67 = (int32_t) ((int64_t) (uint64_t) v66 - (uint64_t) ((int64_t) (uint64_t) v12 % (uint64_t) v66));
              Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v68 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v67 < v11 ? v67 : v11);
              uint64_t v69 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v65 + (v7 + v7 * v5) + v7 * v6));
              TASSIGN(v68, v69);
              __ubuf__ half* v70 = v36.data();
              int32_t v71 = (int32_t) ((int64_t) (uint64_t) v66 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v33) % (uint64_t) v66));
              Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v72 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v71 < v11 ? v71 : v11);
              uint64_t v73 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v70 + (v7 + v7 * v5) + (unsigned) v33 * v6));
              TASSIGN(v72, v73);
              wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
              TLOAD(v36, v64);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
              for (size_t v74 = v18; v74 < v20; v74 += v19) {
                pipe_barrier(PIPE_V);
                TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v37, v36);
                TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v38, v36);
                pipe_barrier(PIPE_V);
                TADD(v68, v37, v38);
                pipe_barrier(PIPE_V);
                TSUB(v72, v37, v38);
              };
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
              pipe_barrier(PIPE_MTE3);
              TSTORE(v64, v36);
              set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
            };
          };
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        };
      };
    };
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  #endif // __DAV_VEC__

  return;
}

