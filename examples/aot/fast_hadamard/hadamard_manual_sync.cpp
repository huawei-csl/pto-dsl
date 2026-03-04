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
  int64_t v13 = 49152;
  int64_t v14 = 16384;
  int64_t v15 = 32768;
  int64_t v16 = 81920;
  int64_t v17 = 0;
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
          int32_t v37 = v3 < v8 ? v8 / v3 : v10;
          int32_t v38 = v28 / v37;
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
          for (size_t v39 = v19; v39 < ((size_t) (v28 % v37 != v11 && v28 < v11 == v37 < v11 ? v38 + v10 : v38)); v39 += v20) {
            int32_t v40 = (int32_t) v39;
            int32_t v41 = (int32_t) ((uint32_t) v40 * (uint32_t) v37);
            int32_t v42 = (int32_t) ((uint32_t) v28 - (uint32_t) v41);
            int32_t v43 = v42 < v37 ? v42 : v37;
            size_t v44 = (size_t) v43;
            if (v43 > v11) {
              int32_t v45 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v27 + (uint32_t) v41) * (uint32_t) v3);
              if (v40 % v9 == v11) {
                for (size_t v46 = v19; v46 < v44; v46 += v20) {
                  unsigned v47 = (unsigned) v3 * v6;
                  pto::Shape<1, 1, 1, 1, -1> v48 = pto::Shape<1, 1, 1, 1, -1>(v3);
                  pto::Stride<-1, -1, -1, -1, 1> v49 = pto::Stride<-1, -1, -1, -1, 1>(v47, v47, v47, v47);
                  GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v50 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v45 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v46) * (uint32_t) v3)) * (unsigned) v10), v48, v49);
                  __ubuf__ half* v51 = v30.data();
                  int64_t v52 = (int64_t) v3;
                  int32_t v53 = (int32_t) ((int64_t) (uint64_t) v52 - (uint64_t) ((int64_t) (uint64_t) v17 % (uint64_t) v52));
                  Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v54 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v53 < v12 ? v53 : v12);
                  uint64_t v55 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v51 + (v7 + v7 * v5) + v7 * v6));
                  TASSIGN(v54, v55);
                  __ubuf__ half* v56 = v30.data();
                  int32_t v57 = (int32_t) ((int64_t) (uint64_t) v52 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v31) % (uint64_t) v52));
                  Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v58 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v57 < v12 ? v57 : v12);
                  uint64_t v59 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v56 + (v7 + v7 * v5) + (unsigned) v31 * v6));
                  TASSIGN(v58, v59);
                  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                  TLOAD(v30, v50);
                  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                  for (size_t v60 = v19; v60 < v21; v60 += v20) {
                    TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v32, v30);
                    TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v33, v30);
                    pipe_barrier(PIPE_V);
                    TADD(v54, v32, v33);
                    TSUB(v58, v32, v33);
                    pipe_barrier(PIPE_V);
                  };
                  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                  TSTORE(v50, v30);
                  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                };
              } else {
                for (size_t v61 = v19; v61 < v44; v61 += v20) {
                  unsigned v62 = (unsigned) v3 * v6;
                  pto::Shape<1, 1, 1, 1, -1> v63 = pto::Shape<1, 1, 1, 1, -1>(v3);
                  pto::Stride<-1, -1, -1, -1, 1> v64 = pto::Stride<-1, -1, -1, -1, 1>(v62, v62, v62, v62);
                  GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v65 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v45 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v61) * (uint32_t) v3)) * (unsigned) v10), v63, v64);
                  __ubuf__ half* v66 = v34.data();
                  int64_t v67 = (int64_t) v3;
                  int32_t v68 = (int32_t) ((int64_t) (uint64_t) v67 - (uint64_t) ((int64_t) (uint64_t) v17 % (uint64_t) v67));
                  Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v69 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v68 < v12 ? v68 : v12);
                  uint64_t v70 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v66 + (v7 + v7 * v5) + v7 * v6));
                  TASSIGN(v69, v70);
                  __ubuf__ half* v71 = v34.data();
                  int32_t v72 = (int32_t) ((int64_t) (uint64_t) v67 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v31) % (uint64_t) v67));
                  Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v73 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v72 < v12 ? v72 : v12);
                  uint64_t v74 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v71 + (v7 + v7 * v5) + (unsigned) v31 * v6));
                  TASSIGN(v73, v74);
                  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                  TLOAD(v34, v65);
                  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                  for (size_t v75 = v19; v75 < v21; v75 += v20) {
                    TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v35, v34);
                    TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v36, v34);
                    pipe_barrier(PIPE_V);
                    TADD(v69, v35, v36);
                    TSUB(v73, v35, v36);
                    pipe_barrier(PIPE_V);
                  };
                  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                  TSTORE(v65, v34);
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

