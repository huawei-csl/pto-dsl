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

__global__ AICORE void matmul_kernel_step1_baseline(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, int32_t v4, int32_t v5, int32_t v6) {
  unsigned v7 = 128;
  unsigned v8 = 0;
  const int32_t v9 = 0;
  const int32_t v10 = 1;
  const int32_t v11 = 128;
  const int32_t v12 = 256;
  const int32_t v13 = 512;
  const int32_t v14 = 64;
  const int32_t v15 = 192;
  const int32_t v16 = 320;
  const int32_t v17 = 384;
  const int32_t v18 = 448;
  const int64_t v19 = 0;
  const int64_t v20 = 131072;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v21 = get_block_num();
  int64_t v22 = get_block_idx();
  int32_t v23 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v5 + (uint32_t) v12) - (uint32_t) v10) / v12;
  int32_t v24 = v6 / v13;
  Tile<TileType::Mat, half, 128, 512, BLayout::ColMajor, 128, 512, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v25;
  TASSIGN(v25, v19);
  Tile<TileType::Mat, half, 256, 256, BLayout::RowMajor, 256, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v26;
  TASSIGN(v26, v20);
  Tile<TileType::Left, half, 128, 64, BLayout::RowMajor, 128, 64, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v27;
  TASSIGN(v27, v19);
  Tile<TileType::Right, half, 64, 256, BLayout::RowMajor, 64, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v28;
  TASSIGN(v28, v19);
  Tile<TileType::Acc, float, 128, 256, BLayout::ColMajor, 128, 256, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v29;
  TASSIGN(v29, v19);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v30 = (size_t) ((int32_t) (int64_t) v22); v30 < ((size_t) ((int32_t) (uint32_t) v23 * (uint32_t) (v4 / v11))); v30 += (size_t) ((int32_t) (int64_t) v21)) {
    int32_t v31 = (int32_t) v30;
    int32_t v32 = (int32_t) ((uint32_t) (v31 / v23) * (uint32_t) v11);
    int32_t v33 = (int32_t) ((uint32_t) (v31 % v23) * (uint32_t) v12);
    unsigned v34 = (unsigned) v6;
    unsigned v35 = v7 * v34;
    pto::Shape<1, 1, 1, 128, 512> v36 = pto::Shape<1, 1, 1, 128, 512>();
    pto::Stride<-1, -1, -1, -1, 1> v37 = pto::Stride<-1, -1, -1, -1, 1>(v35, v35, v35, v34);
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v8 + (unsigned) v32 * (unsigned) v6 + v8 * (unsigned) v10), v36, v37);
    pipe_barrier(PIPE_MTE2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v25, v38);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    for (size_t v39 = (size_t) v9; v39 < ((size_t) v24); v39 += (size_t) v10) {
      int32_t v40 = (int32_t) v39;
      int32_t v41 = (int32_t) ((uint32_t) v40 * (uint32_t) v13);
      pto::Shape<1, 1, 1, 256, 256> v42 = pto::Shape<1, 1, 1, 256, 256>();
      pto::Stride<256, 256, 256, 1, -1> v43 = pto::Stride<256, 256, 256, 1, -1>((unsigned) v6);
      GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN>(v2 + (v8 + (unsigned) v41 * (unsigned) v10 + (unsigned) v33 * (unsigned) v6), v42, v43);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v26, v44);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      TEXTRACT(v27, v25, v9, v9);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
      TEXTRACT(v28, v26, v9, v9);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v40 == v9) {
        TMATMUL(v29, v27, v28);
      } else {
        TMATMUL_ACC(v29, v29, v27, v28);
      };
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      pipe_barrier(PIPE_MTE1);
      TEXTRACT(v27, v25, v9, v14);
      TEXTRACT(v28, v26, v14, v9);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL_ACC(v29, v29, v27, v28);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
      TEXTRACT(v27, v25, v9, v11);
      TEXTRACT(v28, v26, v11, v9);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
      TMATMUL_ACC(v29, v29, v27, v28);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
      TEXTRACT(v27, v25, v9, v15);
      TEXTRACT(v28, v26, v15, v9);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
      TMATMUL_ACC(v29, v29, v27, v28);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      pto::Shape<1, 1, 1, 256, 256> v45 = pto::Shape<1, 1, 1, 256, 256>();
      pto::Stride<256, 256, 256, 1, -1> v46 = pto::Stride<256, 256, 256, 1, -1>((unsigned) v6);
      GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 256, 256>, pto::Stride<256, 256, 256, 1, -1>, pto::Layout::DN>(v2 + (v8 + (unsigned) ((int32_t) (uint32_t) v41 + (uint32_t) v12) * (unsigned) v10 + (unsigned) v33 * (unsigned) v6), v45, v46);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
      TLOAD(v26, v47);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
      TEXTRACT(v27, v25, v9, v12);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
      TEXTRACT(v28, v26, v9, v9);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      TMATMUL_ACC(v29, v29, v27, v28);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
      TEXTRACT(v27, v25, v9, v16);
      TEXTRACT(v28, v26, v14, v9);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
      TMATMUL_ACC(v29, v29, v27, v28);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
      TEXTRACT(v27, v25, v9, v17);
      TEXTRACT(v28, v26, v11, v9);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
      TMATMUL_ACC(v29, v29, v27, v28);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
      TEXTRACT(v27, v25, v9, v18);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
      TEXTRACT(v28, v26, v15, v9);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
      TMATMUL_ACC(v29, v29, v27, v28);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
      if ((int32_t) ((uint32_t) v40 + (uint32_t) v10) < v24) {
        unsigned v48 = (unsigned) v6;
        unsigned v49 = v7 * v48;
        pto::Shape<1, 1, 1, 128, 512> v50 = pto::Shape<1, 1, 1, 128, 512>();
        pto::Stride<-1, -1, -1, -1, 1> v51 = pto::Stride<-1, -1, -1, -1, 1>(v49, v49, v49, v48);
        GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v52 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 512>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v8 + (unsigned) v32 * (unsigned) v6 + (unsigned) ((int32_t) (uint32_t) v41 + (uint32_t) v13) * (unsigned) v10), v50, v51);
        TLOAD(v25, v52);
      };
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    unsigned v53 = (unsigned) v5;
    unsigned v54 = v7 * v53;
    pto::Shape<1, 1, 1, 128, 256> v55 = pto::Shape<1, 1, 1, 128, 256>();
    pto::Stride<-1, -1, -1, -1, 1> v56 = pto::Stride<-1, -1, -1, -1, 1>(v54, v54, v54, v53);
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v57 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 256>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v3 + (v8 + (unsigned) v32 * (unsigned) v5 + (unsigned) v33 * (unsigned) v10), v55, v56);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v57, v29);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
