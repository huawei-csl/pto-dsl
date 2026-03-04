#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void _kernel(__gm__ half* v1, int32_t v2, int32_t v3, int32_t v4) {
  unsigned v5 = 16384;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 16384;
  int32_t v9 = 2;
  int32_t v10 = 1;
  int32_t v11 = 0;
  int32_t v12 = 8192;
  int64_t v13 = 0;
  int64_t v14 = 32768;
  int64_t v15 = 49152;
  int64_t v16 = 65536;
  int64_t v17 = 98304;
  int64_t v18 = 114688;
  using T = float;
  size_t v19 = (size_t) v11;
  size_t v20 = (size_t) v10;
  size_t v21 = (size_t) v4;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v22 = get_block_idx();
  int64_t v23 = get_block_num();
  int32_t v24 = (int32_t) ((int64_t) v23);
  int32_t v25 = v2 / v24;
  int32_t v26 = v2 % v24 != v11 && v2 < v11 == v24 < v11 ? v25 + v10 : v25;
  int32_t v27 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v22) * (uint32_t) v26);
  if (v3 > v11) {
    if (v3 <= v8) {
      if (v27 < v2) {
        int32_t v28 = (int32_t) ((uint32_t) v27 + (uint32_t) v26) > v2 ? (int32_t) ((uint32_t) v2 - (uint32_t) v27) : v26;
        if (v28 > v11) {
          int32_t v29 = (int32_t) ((uint32_t) v2 * (uint32_t) v3);
          Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v30 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v3);
          TASSIGN(v30, v13);
          int32_t v31 = v3 / v9;
          Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v32 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v31);
          TASSIGN(v32, v14);
          Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v33 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v31);
          TASSIGN(v33, v15);
          Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v34 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v3);
          TASSIGN(v34, v16);
          Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v35 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v31);
          TASSIGN(v35, v17);
          Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v36 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v31);
          TASSIGN(v36, v18);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
          for (size_t v37 = v19; v37 < ((size_t) v28); v37 += v20) {
            int32_t v38 = (int32_t) v37;
            int32_t v39 = (int32_t) ((uint32_t) v28 - (uint32_t) v38);
            int32_t v40 = v39 < v10 ? v39 : v10;
            size_t v41 = (size_t) v40;
            if (v40 > v11) {
              int32_t v42 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v27 + (uint32_t) v38) * (uint32_t) v3);
              if (v38 % v9 == v11) {
                for (size_t v43 = v19; v43 < v41; v43 += v20) {
                  unsigned v44 = (unsigned) v3 * v6;
                  pto::Shape<1, 1, 1, 1, -1> v45 = pto::Shape<1, 1, 1, 1, -1>(v3);
                  pto::Stride<-1, -1, -1, -1, 1> v46 = pto::Stride<-1, -1, -1, -1, 1>(v44, v44, v44, v44);
                  GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v42 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v43) * (uint32_t) v3)) * (unsigned) v10), v45, v46);
                  __ubuf__ half* v48 = v30.data();
                  int64_t v49 = (int64_t) v3;
                  int32_t v50 = (int32_t) ((int64_t) (uint64_t) v49 - (uint64_t) ((int64_t) (uint64_t) v13 % (uint64_t) v49));
                  Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v51 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v50 < v12 ? v50 : v12);
                  uint64_t v52 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v48 + (v7 + v7 * v5) + v7 * v6));
                  TASSIGN(v51, v52);
                  __ubuf__ half* v53 = v30.data();
                  int32_t v54 = (int32_t) ((int64_t) (uint64_t) v49 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v31) % (uint64_t) v49));
                  Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v55 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v54 < v12 ? v54 : v12);
                  uint64_t v56 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v53 + (v7 + v7 * v5) + (unsigned) v31 * v6));
                  TASSIGN(v55, v56);
                  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                  TLOAD(v30, v47);
                  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                  for (size_t v57 = v19; v57 < v21; v57 += v20) {
                    TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v32, v30);
                    TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v33, v30);
                    pipe_barrier(PIPE_V);
                    TADD(v51, v32, v33);
                    TSUB(v55, v32, v33);
                    pipe_barrier(PIPE_V);
                  };
                  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                  TSTORE(v47, v30);
                  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                };
              } else {
                for (size_t v58 = v19; v58 < v41; v58 += v20) {
                  unsigned v59 = (unsigned) v3 * v6;
                  pto::Shape<1, 1, 1, 1, -1> v60 = pto::Shape<1, 1, 1, 1, -1>(v3);
                  pto::Stride<-1, -1, -1, -1, 1> v61 = pto::Stride<-1, -1, -1, -1, 1>(v59, v59, v59, v59);
                  GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v62 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v42 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v58) * (uint32_t) v3)) * (unsigned) v10), v60, v61);
                  __ubuf__ half* v63 = v34.data();
                  int64_t v64 = (int64_t) v3;
                  int32_t v65 = (int32_t) ((int64_t) (uint64_t) v64 - (uint64_t) ((int64_t) (uint64_t) v13 % (uint64_t) v64));
                  Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v66 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v65 < v12 ? v65 : v12);
                  uint64_t v67 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v63 + (v7 + v7 * v5) + v7 * v6));
                  TASSIGN(v66, v67);
                  __ubuf__ half* v68 = v34.data();
                  int32_t v69 = (int32_t) ((int64_t) (uint64_t) v64 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v31) % (uint64_t) v64));
                  Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v70 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v69 < v12 ? v69 : v12);
                  uint64_t v71 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v68 + (v7 + v7 * v5) + (unsigned) v31 * v6));
                  TASSIGN(v70, v71);
                  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                  TLOAD(v34, v62);
                  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                  for (size_t v72 = v19; v72 < v21; v72 += v20) {
                    TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v35, v34);
                    TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v36, v34);
                    pipe_barrier(PIPE_V);
                    TADD(v66, v35, v36);
                    TSUB(v70, v35, v36);
                    pipe_barrier(PIPE_V);
                  };
                  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                  TSTORE(v62, v34);
                  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                };
              };
            };
          };
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        };
      };
    };
  }
  #endif // __DAV_VEC__

  return;
}

