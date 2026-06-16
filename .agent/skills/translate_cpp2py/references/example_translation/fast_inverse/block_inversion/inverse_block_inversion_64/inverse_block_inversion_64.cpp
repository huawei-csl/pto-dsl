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

__global__ AICORE void tri_inv_block2x2_fp16(__gm__ float* v1, __gm__ half* v2, __gm__ half* v3, int32_t v4) {
  unsigned v5 = 32;
  unsigned v6 = 0;
  const int32_t v7 = 0;
  const int32_t v8 = 1;
  const int32_t v9 = 64;
  const int32_t v10 = 32;
  const int64_t v11 = 4096;
  const int64_t v12 = 8192;
  const int64_t v13 = 6144;
  const int64_t v14 = 14336;
  const int64_t v15 = 10240;
  const int64_t v16 = 0;
  const int64_t v17 = 2048;
  const int64_t v18 = 12288;
  using T = float;
  size_t v19 = (size_t) v8;
  size_t v20 = (size_t) v7;

  #if defined(__DAV_CUBE__)
  int32_t v21 = (int32_t) ((uint32_t) v4 - (uint32_t) v8);
  size_t v22 = (size_t) v21;
  int64_t v23 = get_block_idx();
  int64_t v24 = get_block_num();
  int32_t v25 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v24) * (uint32_t) v9);
  int32_t v26 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v23) * (uint32_t) v9);
  int32_t v27 = (int32_t) ((uint32_t) v26 + (uint32_t) v10);
  pto::Shape<1, 1, 1, 32, 32> v28 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v29 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v30 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v3 + (v6 + v6 * (unsigned) v10 + v6 * (unsigned) v8), v28, v29);
  pto::Shape<1, 1, 1, 32, 32> v31 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<2048, 2048, 2048, 64, 1> v32 = pto::Stride<2048, 2048, 2048, 64, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v33 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v26 * (unsigned) v9 + v6 * (unsigned) v8), v31, v32);
  pto::Shape<1, 1, 1, 32, 32> v34 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<2048, 2048, 2048, 64, 1> v35 = pto::Stride<2048, 2048, 2048, 64, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v36 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v27 * (unsigned) v9 + v6 * (unsigned) v8), v34, v35);
  pto::Shape<1, 1, 1, 32, 32> v37 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<2048, 2048, 2048, 64, 1> v38 = pto::Stride<2048, 2048, 2048, 64, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v39 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v2 + (v6 + (unsigned) v27 * (unsigned) v9 + v5 * (unsigned) v8), v37, v38);
  pto::Shape<1, 1, 1, 32, 32> v40 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<2048, 2048, 2048, 64, 1> v41 = pto::Stride<2048, 2048, 2048, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v26 * (unsigned) v9 + v6 * (unsigned) v8), v40, v41);
  pto::Shape<1, 1, 1, 32, 32> v43 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<2048, 2048, 2048, 64, 1> v44 = pto::Stride<2048, 2048, 2048, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v27 * (unsigned) v9 + v6 * (unsigned) v8), v43, v44);
  pto::Shape<1, 1, 1, 32, 32> v46 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<2048, 2048, 2048, 64, 1> v47 = pto::Stride<2048, 2048, 2048, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<2048, 2048, 2048, 64, 1>, pto::Layout::ND>(v1 + (v6 + (unsigned) v27 * (unsigned) v9 + v5 * (unsigned) v8), v46, v47);
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v49;
  TASSIGN(v49, v11);
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v50;
  TASSIGN(v50, v12);
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v51;
  TASSIGN(v51, v13);
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v52;
  TASSIGN(v52, v14);
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v53;
  TASSIGN(v53, v15);
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v54;
  TASSIGN(v54, v16);
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v55;
  TASSIGN(v55, v17);
  Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v56;
  TASSIGN(v56, v18);
  Tile<TileType::Left, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v57;
  TASSIGN(v57, v16);
  Tile<TileType::Right, half, 32, 32, BLayout::RowMajor, 32, 32, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v58;
  TASSIGN(v58, v16);
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v59;
  TASSIGN(v59, v16);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  TLOAD(v54, v30);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v57, v54);
  TMOV(v58, v54);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v59, v57, v58);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TMOV(v55, v59);
  TMOV(v49, v59);
  TMOV(v51, v59);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  TLOAD(v50, v33);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  TMOV(v57, v50);
  TMOV(v58, v54);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  TMATMUL(v59, v57, v58);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TMOV(v50, v59);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  for (size_t v60 = v20; v60 < v22; v60 += v19) {
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    TMOV(v57, v49);
    TMOV(v58, v55);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    TMATMUL(v59, v57, v58);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    TMOV(v58, v50);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    TMATMUL_ACC(v59, v59, v57, v58);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    if ((int32_t) ((uint32_t) ((int32_t) v60) + (uint32_t) v8) < v21) {
      TMOV(v49, v59);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
      TMOV(v57, v50);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID3);
      TMATMUL(v59, v57, v58);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
      TMOV(v50, v59);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
  }
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
  TMOV(v49, v59);
  TSTORE(v42, v59);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
  TLOAD(v52, v39);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID6);
  TMOV(v57, v52);
  TMOV(v58, v54);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID4);
  TMATMUL(v59, v57, v58);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID5);
  TMOV(v52, v59);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID5);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID5);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  for (size_t v61 = v20; v61 < v22; v61 += v19) {
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    TMOV(v57, v51);
    TMOV(v58, v55);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID6);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
    TMATMUL(v59, v57, v58);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    TMOV(v58, v52);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
    TMATMUL_ACC(v59, v59, v57, v58);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    if ((int32_t) ((uint32_t) ((int32_t) v61) + (uint32_t) v8) < v21) {
      TMOV(v51, v59);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID7);
      TMOV(v57, v52);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID7);
      TMATMUL(v59, v57, v58);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID7);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID7);
      TMOV(v52, v59);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TMOV(v51, v59);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  TSTORE(v48, v59);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  TLOAD(v53, v36);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
  TMOV(v57, v51);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
  TMOV(v58, v53);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  TMATMUL(v59, v57, v58);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TMOV(v56, v59);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID1);
  TMOV(v57, v56);
  TMOV(v58, v49);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v59, v57, v58);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TMOV(v56, v59);
  set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  TMOV(v57, v54);
  wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID2);
  TMOV(v58, v56);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  TMATMUL(v59, v57, v58);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v45, v59);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID6);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
