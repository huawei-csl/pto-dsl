#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void tri_inv_trick_fp16(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, int32_t v4, int32_t v5) {
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 1;
  int32_t v9 = 2;
  int32_t v10 = 4;
  int32_t v11 = 8;
  int32_t v12 = 128;
  int64_t v13 = 32768;
  int64_t v14 = 0;
  int64_t v15 = 65536;
  using T = float;
  size_t v16 = (size_t) v8;

  #if defined(__DAV_CUBE__)
  int64_t v17 = get_block_idx();
  int64_t v18 = get_block_num();
  if (v4 <= v12) {
    int32_t v19 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v18) * (uint32_t) v4);
    int32_t v20 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v17) * (uint32_t) v4);
    unsigned v21 = (unsigned) v4;
    unsigned v22 = (unsigned) v4;
    unsigned v23 = (unsigned) v4 * v22;
    pto::Shape<1, 1, 1, -1, -1> v24 = pto::Shape<1, 1, 1, -1, -1>(v4, v4);
    pto::Stride<-1, -1, -1, -1, 1> v25 = pto::Stride<-1, -1, -1, -1, 1>(v23, v23, v23, v22);
    GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v26 = GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v20 * (unsigned) v4 + v7 * (unsigned) v8), v24, v25);
    unsigned v27 = (unsigned) v4;
    unsigned v28 = (unsigned) v4;
    unsigned v29 = (unsigned) v4 * v28;
    pto::Shape<1, 1, 1, -1, -1> v30 = pto::Shape<1, 1, 1, -1, -1>(v4, v4);
    pto::Stride<-1, -1, -1, -1, 1> v31 = pto::Stride<-1, -1, -1, -1, 1>(v29, v29, v29, v28);
    GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v32 = GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v3 + (v7 + v7 * (unsigned) v4 + v7 * (unsigned) v8), v30, v31);
    unsigned v33 = (unsigned) v4;
    unsigned v34 = (unsigned) v4;
    unsigned v35 = (unsigned) v4 * v34;
    pto::Shape<1, 1, 1, -1, -1> v36 = pto::Shape<1, 1, 1, -1, -1>(v4, v4);
    pto::Stride<-1, -1, -1, -1, 1> v37 = pto::Stride<-1, -1, -1, -1, 1>(v35, v35, v35, v34);
    GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v20 * (unsigned) v4 + v7 * (unsigned) v8), v36, v37);
    Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null> v39 = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null>(v4, v4);
    TASSIGN(v39, v13);
    Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null> v40 = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null>(v4, v4);
    TASSIGN(v40, v14);
    Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null> v41 = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null>(v4, v4);
    TASSIGN(v41, v15);
    Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null> v42 = Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null>(v4, v4);
    TASSIGN(v42, v14);
    Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null> v43 = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null>(v4, v4);
    TASSIGN(v43, v14);
    Tile<TileType::Acc, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null> v44 = Tile<TileType::Acc, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null>(v4, v4);
    TASSIGN(v44, v14);
    TLOAD(v40, v26);
    TLOAD(v39, v32);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v42, v40);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMOV(v43, v40);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(v44, v42, v43);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v40, v44);
    TMOV(v43, v39);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(v44, v42, v43);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v42, v39);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL_ACC(v44, v44, v42, v43);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v39, v44);
    TMATMUL(v44, v42, v43);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    TMOV(v41, v44);
    for (size_t v45 = v16; v45 < ((size_t) v5); v45 += v16) {
      int32_t v46 = (int32_t) v45;
      if (v46 == v8) {
        TMOV(v42, v39);
        TMOV(v43, v41);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL(v44, v42, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TMOV(v43, v40);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v42, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        if (v46 < v5 / v9) {
          TMOV(v39, v44);
          TMOV(v42, v40);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          TMATMUL(v44, v42, v43);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          TMOV(v40, v44);
        };
      } else {
        if (v46 == v9) {
          TMOV(v42, v39);
          TMOV(v43, v41);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          TMATMUL(v44, v42, v43);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          TMOV(v43, v40);
          set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
          TMATMUL_ACC(v44, v44, v42, v43);
          set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
          if (v46 < v5 / v9) {
            TMOV(v39, v44);
            TMOV(v42, v40);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            TMATMUL(v44, v42, v43);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            TMOV(v40, v44);
          };
        } else {
          if (v46 == v10) {
            TMOV(v42, v39);
            TMOV(v43, v41);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            TMATMUL(v44, v42, v43);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            TMOV(v43, v40);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            TMATMUL_ACC(v44, v44, v42, v43);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            if (v46 < v5 / v9) {
              TMOV(v39, v44);
              TMOV(v42, v40);
              set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
              wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
              TMATMUL(v44, v42, v43);
              set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
              wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
              TMOV(v40, v44);
            };
          } else {
            if (v46 == v11) {
              TMOV(v42, v39);
              TMOV(v43, v41);
              set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
              wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
              TMATMUL(v44, v42, v43);
              set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
              wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
              TMOV(v43, v40);
              set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
              wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
              TMATMUL_ACC(v44, v44, v42, v43);
              set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
              wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
              if (v46 < v5 / v9) {
                TMOV(v39, v44);
                TMOV(v42, v40);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                TMATMUL(v44, v42, v43);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                TMOV(v40, v44);
              };
            };
          };
        };
      };
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v38, v44);
  }
  #endif // __DAV_CUBE__

  return;
}

