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
  int64_t v13 = 32768;
  int64_t v14 = 65536;
  int64_t v15 = 0;
  int64_t v16 = 81920;
  int64_t v17 = 114688;
  int64_t v18 = 16384;
  using T = float;
  size_t v19 = (size_t) v11;
  size_t v20 = (size_t) v10;
  size_t v21 = (size_t) v4;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  if (v3 > v11) {
    if (v3 <= v8) {
      int64_t v22 = get_block_idx();
      int64_t v23 = get_block_num();
      int32_t v24 = (int32_t) ((int64_t) v23);
      int32_t v25 = v2 / v24;
      int32_t v26 = v2 % v24 != v11 && v2 < v11 == v24 < v11 ? v25 + v10 : v25;
      int32_t v27 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v22) * (uint32_t) v26);
      int32_t v28 = (int32_t) ((uint32_t) v2 * (uint32_t) v3);
      Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v29 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v3);
      TASSIGN(v29, v13);
      int32_t v30 = v3 / v9;
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v31 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v30);
      TASSIGN(v31, v14);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v32 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v30);
      TASSIGN(v32, v15);
      Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v33 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v3);
      TASSIGN(v33, v16);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v34 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v30);
      TASSIGN(v34, v17);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v35 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v30);
      TASSIGN(v35, v18);
      if (v27 < v2) {
        int32_t v36 = (int32_t) ((uint32_t) v27 + (uint32_t) v26);
        int32_t v37 = (int32_t) ((uint32_t) ((uint32_t) v36 < (uint32_t) v2 ? v36 : v2) - (uint32_t) v27);
        int32_t v38 = v3 < v8 ? v8 / v3 : v10;
        int32_t v39 = v37 / v38;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        for (size_t v40 = v19; v40 < ((size_t) (v37 % v38 != v11 && v37 < v11 == v38 < v11 ? v39 + v10 : v39)); v40 += v20) {
          int32_t v41 = (int32_t) v40;
          int32_t v42 = (int32_t) ((uint32_t) v41 * (uint32_t) v38);
          int32_t v43 = (int32_t) ((uint32_t) v37 - (uint32_t) v42);
          int32_t v44 = v43 < v38 ? v43 : v38;
          size_t v45 = (size_t) v44;
          if (v44 > v11) {
            int32_t v46 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v27 + (uint32_t) v42) * (uint32_t) v3);
            if (v41 % v9 == v11) {
              for (size_t v47 = v19; v47 < v45; v47 += v20) {
                unsigned v48 = (unsigned) v3 * v6;
                pto::Shape<1, 1, 1, 1, -1> v49 = pto::Shape<1, 1, 1, 1, -1>(v3);
                pto::Stride<-1, -1, -1, -1, 1> v50 = pto::Stride<-1, -1, -1, -1, 1>(v48, v48, v48, v48);
                GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v51 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v46 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v47) * (uint32_t) v3)) * (unsigned) v10), v49, v50);
                __ubuf__ half* v52 = v29.data();
                int64_t v53 = (int64_t) v3;
                int32_t v54 = (int32_t) ((int64_t) (uint64_t) v53 - (uint64_t) ((int64_t) (uint64_t) v15 % (uint64_t) v53));
                Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v55 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v54 < v12 ? v54 : v12);
                uint64_t v56 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v52 + (v7 + v7 * v5) + v7 * v6));
                TASSIGN(v55, v56);
                __ubuf__ half* v57 = v29.data();
                int32_t v58 = (int32_t) ((int64_t) (uint64_t) v53 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v30) % (uint64_t) v53));
                Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v59 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v58 < v12 ? v58 : v12);
                uint64_t v60 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v57 + (v7 + v7 * v5) + (unsigned) v30 * v6));
                TASSIGN(v59, v60);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                TLOAD(v29, v51);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                for (size_t v61 = v19; v61 < v21; v61 += v20) {
                  TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v31, v29);
                  TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v32, v29);
                  pipe_barrier(PIPE_V);
                  TADD(v55, v31, v32);
                  TSUB(v59, v31, v32);
                  pipe_barrier(PIPE_V);
                };
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                TSTORE(v51, v29);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
              };
            } else {
              for (size_t v62 = v19; v62 < v45; v62 += v20) {
                unsigned v63 = (unsigned) v3 * v6;
                pto::Shape<1, 1, 1, 1, -1> v64 = pto::Shape<1, 1, 1, 1, -1>(v3);
                pto::Stride<-1, -1, -1, -1, 1> v65 = pto::Stride<-1, -1, -1, -1, 1>(v63, v63, v63, v63);
                GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v66 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v46 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v62) * (uint32_t) v3)) * (unsigned) v10), v64, v65);
                __ubuf__ half* v67 = v33.data();
                int64_t v68 = (int64_t) v3;
                int32_t v69 = (int32_t) ((int64_t) (uint64_t) v68 - (uint64_t) ((int64_t) (uint64_t) v15 % (uint64_t) v68));
                Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v70 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v69 < v12 ? v69 : v12);
                uint64_t v71 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v67 + (v7 + v7 * v5) + v7 * v6));
                TASSIGN(v70, v71);
                __ubuf__ half* v72 = v33.data();
                int32_t v73 = (int32_t) ((int64_t) (uint64_t) v68 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v30) % (uint64_t) v68));
                Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null> v74 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>(v73 < v12 ? v73 : v12);
                uint64_t v75 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v72 + (v7 + v7 * v5) + (unsigned) v30 * v6));
                TASSIGN(v74, v75);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                TLOAD(v33, v66);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                for (size_t v76 = v19; v76 < v21; v76 += v20) {
                  TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P0101>(v34, v33);
                  TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null>, MaskPattern::P1010>(v35, v33);
                  pipe_barrier(PIPE_V);
                  TADD(v70, v34, v35);
                  TSUB(v74, v34, v35);
                  pipe_barrier(PIPE_V);
                };
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TSTORE(v66, v33);
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
  }
  #endif // __DAV_VEC__

  return;
}

