#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

__global__ AICORE void matmul_kernel_ABt(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, int32_t v4, int32_t v5, int32_t v6) {
  unsigned v7 = 128;
  unsigned v8 = 0;
  const int32_t v9 = 0;
  const int32_t v10 = 1;
  const int32_t v11 = 2;
  const int32_t v12 = 128;
  const int32_t v13 = 256;
  const int32_t v14 = 512;
  const int32_t v15 = 5;
  const int32_t v16 = 64;
  const int32_t v17 = 192;
  const int32_t v18 = 320;
  const int32_t v19 = 384;
  const int32_t v20 = 448;
  const int32_t v21 = 4;
  const int64_t v22 = 0;
  const int64_t v23 = 393216;
  const int64_t v24 = 262144;
  const int64_t v25 = 131072;
  const int64_t v26 = 16384;
  const int64_t v27 = 32768;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v28 = get_block_num();
  int32_t v29 = (int32_t) ((int64_t) v28);
  int64_t v30 = get_block_idx();
  int32_t v31 = (int32_t) ((int64_t) v30);
  int32_t v32 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v5 + (uint32_t) v13) - (uint32_t) v10) / v13;
  int32_t v33 = v4 / v12;
  int32_t v34 = (int32_t) ((uint32_t) v32 * (uint32_t) v33);
  int32_t v35 = v6 / v14;
  Tile<TileType::Mat, half, 128, 512, BLayout::ColMajor, 128, 512, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v36;
  TASSIGN(v36, v22);
  Tile<TileType::Mat, half, 128, 512, BLayout::ColMajor, 128, 512, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v37;
  TASSIGN(v37, v23);
  Tile<TileType::Mat, half, 256, 256, BLayout::RowMajor, 256, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v38;
  TASSIGN(v38, v24);
  Tile<TileType::Mat, half, 256, 256, BLayout::RowMajor, 256, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v39;
  TASSIGN(v39, v25);
  Tile<TileType::Left, half, 128, 64, BLayout::RowMajor, 128, 64, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v40;
  TASSIGN(v40, v22);
  Tile<TileType::Left, half, 128, 64, BLayout::RowMajor, 128, 64, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v41;
  TASSIGN(v41, v26);
  Tile<TileType::Right, half, 64, 256, BLayout::RowMajor, 64, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v42;
  TASSIGN(v42, v27);
  Tile<TileType::Right, half, 64, 256, BLayout::RowMajor, 64, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v43;
  TASSIGN(v43, v22);
  Tile<TileType::Acc, float, 128, 256, BLayout::ColMajor, 128, 256, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v44;
  TASSIGN(v44, v22);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  for (size_t v45 = (size_t) v31; v45 < ((size_t) v34); v45 += (size_t) v29) {
    int32_t v46 = (int32_t) v45;
    int32_t v47 = (int32_t) ((uint32_t) v33 * (uint32_t) v15);
    int32_t v48 = v46 / v47;
    int32_t v49 = v46 % v47;
    int32_t v50 = (int32_t) ((uint32_t) v48 * (uint32_t) v15);
    int32_t v51 = v48 == (int32_t) ((uint32_t) ((int32_t) ((uint32_t) v32 + (uint32_t) v21) / v15) - (uint32_t) v10) ? (int32_t) ((uint32_t) v32 - (uint32_t) v50) : v15;
    int32_t v52 = v49 / v51;
    int32_t v53 = (int32_t) ((uint32_t) (v48 % v11 == v10 ? (int32_t) ((uint32_t) ((int32_t) (uint32_t) v33 - (uint32_t) v52) - (uint32_t) v10) : v52) * (uint32_t) v12);
    int32_t v54 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v50 + (uint32_t) (v49 % v51)) * (uint32_t) v13);
    if (v46 != v31) {
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    };
    unsigned v55 = (unsigned) v6;
    unsigned v56 = v7 * v55;
    pto::Shape<1, 1, 1, 128, 512> v57 = pto::Shape<1, 1, 1, 128, 512>();
    pto::Stride<-1, -1, -1, -1, 1> v58 = pto::Stride<-1, -1, -1, -1, 1>(v56, v56, v56, v55);
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v59 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v8 + (unsigned) v53 * (unsigned) v6 + v8 * (unsigned) v10), v57, v58);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v36, v59);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    for (size_t v60 = (size_t) v9; v60 < ((size_t) v35); v60 += (size_t) v10) {
      int32_t v61 = (int32_t) v60;
      int32_t v62 = (int32_t) ((uint32_t) v61 * (uint32_t) v14);
      if (v61 % v11 == v9) {
        pto::Shape<1, 1, 1, 256, 256> v63 = pto::Shape<1, 1, 1, 256, 256>();
        pto::Stride<256, 256, 256, 1, -1> v64 = pto::Stride<256, 256, 256, 1, -1>((unsigned) v6);
        GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN> v65 = GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN>(v2 + (v8 + (unsigned) v62 * (unsigned) v10 + (unsigned) v54 * (unsigned) v6), v63, v64);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v38, v65);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v40, v36, v9, v9);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(v42, v38, v9, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (v61 == v9) {
          TMATMUL(v44, v40, v42);
        } else {
          TMATMUL_ACC(v44, v44, v40, v42);
        };
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v41, v36, v9, v16);
        TEXTRACT(v43, v38, v16, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v41, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v40, v36, v9, v12);
        TEXTRACT(v42, v38, v12, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v40, v42);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v41, v36, v9, v17);
        TEXTRACT(v43, v38, v17, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v41, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        pto::Shape<1, 1, 1, 256, 256> v66 = pto::Shape<1, 1, 1, 256, 256>();
        pto::Stride<256, 256, 256, 1, -1> v67 = pto::Stride<256, 256, 256, 1, -1>((unsigned) v6);
        GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN> v68 = GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN>(v2 + (v8 + (unsigned) ((int32_t) (uint32_t) v62 + (uint32_t) v13) * (unsigned) v10 + (unsigned) v54 * (unsigned) v6), v66, v67);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        TLOAD(v39, v68);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v40, v36, v9, v13);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        TEXTRACT(v42, v39, v9, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v40, v42);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v41, v36, v9, v18);
        TEXTRACT(v43, v39, v16, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v41, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v40, v36, v9, v19);
        TEXTRACT(v42, v39, v12, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v40, v42);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v41, v36, v9, v20);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        TEXTRACT(v43, v39, v17, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v41, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        if ((int32_t) ((uint32_t) v61 + (uint32_t) v10) < v35) {
          unsigned v69 = (unsigned) v6;
          unsigned v70 = v7 * v69;
          pto::Shape<1, 1, 1, 128, 512> v71 = pto::Shape<1, 1, 1, 128, 512>();
          pto::Stride<-1, -1, -1, -1, 1> v72 = pto::Stride<-1, -1, -1, -1, 1>(v70, v70, v70, v69);
          GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v73 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v8 + (unsigned) v53 * (unsigned) v6 + (unsigned) ((int32_t) (uint32_t) v62 + (uint32_t) v14) * (unsigned) v10), v71, v72);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
          TLOAD(v37, v73);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        };
      } else {
        pto::Shape<1, 1, 1, 256, 256> v74 = pto::Shape<1, 1, 1, 256, 256>();
        pto::Stride<256, 256, 256, 1, -1> v75 = pto::Stride<256, 256, 256, 1, -1>((unsigned) v6);
        GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN> v76 = GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN>(v2 + (v8 + (unsigned) v62 * (unsigned) v10 + (unsigned) v54 * (unsigned) v6), v74, v75);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        TLOAD(v38, v76);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v40, v37, v9, v9);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        TEXTRACT(v42, v38, v9, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        if (v61 == v9) {
          TMATMUL(v44, v40, v42);
        } else {
          TMATMUL_ACC(v44, v44, v40, v42);
        };
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v41, v37, v9, v16);
        TEXTRACT(v43, v38, v16, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v41, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v40, v37, v9, v12);
        TEXTRACT(v42, v38, v12, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v40, v42);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v41, v37, v9, v17);
        TEXTRACT(v43, v38, v17, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v41, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        pto::Shape<1, 1, 1, 256, 256> v77 = pto::Shape<1, 1, 1, 256, 256>();
        pto::Stride<256, 256, 256, 1, -1> v78 = pto::Stride<256, 256, 256, 1, -1>((unsigned) v6);
        GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN> v79 = GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN>(v2 + (v8 + (unsigned) ((int32_t) (uint32_t) v62 + (uint32_t) v13) * (unsigned) v10 + (unsigned) v54 * (unsigned) v6), v77, v78);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        TLOAD(v39, v79);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v40, v37, v9, v13);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
        TEXTRACT(v42, v39, v9, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v40, v42);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v41, v37, v9, v18);
        TEXTRACT(v43, v39, v16, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v41, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        TEXTRACT(v40, v37, v9, v19);
        TEXTRACT(v42, v39, v12, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v40, v42);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TEXTRACT(v41, v37, v9, v20);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TEXTRACT(v43, v39, v17, v9);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        TMATMUL_ACC(v44, v44, v41, v43);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        if ((int32_t) ((uint32_t) v61 + (uint32_t) v10) < v35) {
          unsigned v80 = (unsigned) v6;
          unsigned v81 = v7 * v80;
          pto::Shape<1, 1, 1, 128, 512> v82 = pto::Shape<1, 1, 1, 128, 512>();
          pto::Stride<-1, -1, -1, -1, 1> v83 = pto::Stride<-1, -1, -1, -1, 1>(v81, v81, v81, v80);
          GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v84 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v8 + (unsigned) v53 * (unsigned) v6 + (unsigned) ((int32_t) (uint32_t) v62 + (uint32_t) v14) * (unsigned) v10), v82, v83);
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          TLOAD(v36, v84);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        };
      };
    };
    unsigned v85 = (unsigned) v5;
    unsigned v86 = v7 * v85;
    pto::Shape<1, 1, 1, 128, 256> v87 = pto::Shape<1, 1, 1, 128, 256>();
    pto::Stride<-1, -1, -1, -1, 1> v88 = pto::Stride<-1, -1, -1, -1, 1>(v86, v86, v86, v85);
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v89 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v3 + (v8 + (unsigned) v53 * (unsigned) v5 + (unsigned) v54 * (unsigned) v10), v87, v88);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v89, v44);
    if ((int32_t) ((uint32_t) v46 + (uint32_t) v29) < v34) {
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    };
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  #endif // __DAV_CUBE__

  return;
}
