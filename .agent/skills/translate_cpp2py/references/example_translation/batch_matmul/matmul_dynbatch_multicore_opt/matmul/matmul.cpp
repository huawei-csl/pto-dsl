#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void RunTMATMULSplitK(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, bool v5, int32_t v6) {
  unsigned v7 = 16384;
  unsigned v8 = 128;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 0;
  int32_t v12 = 1;
  int32_t v13 = 128;
  int32_t v14 = 16384;
  int64_t v15 = 65536;
  int64_t v16 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v17 = get_block_num();
  int32_t v18 = (int32_t) ((int64_t) v17);
  int64_t v19 = get_block_idx();
  int32_t v20 = (int32_t) ((int64_t) v19);
  int32_t v21 = v6 / v18;
  int32_t v22 = v6 % v18;
  int32_t v23 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v20 * (uint32_t) v21) + (uint32_t) ((uint32_t) v20 < (uint32_t) v22 ? v20 : v22));
  int32_t v24 = (int32_t) ((uint32_t) v23 + (uint32_t) ((int32_t) (uint32_t) v21 + (uint32_t) (v20 < v22 ? v12 : v11)));
  Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v25;
  TASSIGN(v25, v15);
  Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v26;
  TASSIGN(v26, v16);
  Tile<TileType::Left, float, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v27;
  TASSIGN(v27, v16);
  Tile<TileType::Right, float, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v16);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v29;
  TASSIGN(v29, v16);
  pto::Shape<1, 1, 1, 128, 128> v30 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v31 = pto::Stride<16384, 16384, 16384, 128, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v32 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v10 + v10 * (unsigned) v13 + v10 * (unsigned) v12), v30, v31);
  TLOAD(v26, v32);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v28, v26);
  for (size_t v33 = (size_t) v23; v33 < ((size_t) ((uint32_t) v24 < (uint32_t) v6 ? v24 : v6)); v33 += (size_t) v12) {
    int32_t v34 = (int32_t) v33;
    pto::Shape<1, 1, 1, 128, 128> v35 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v36 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + ((v10 + (unsigned) v34 * (unsigned) v14) + v10 * (unsigned) v13 + v10 * (unsigned) v12), v35, v36);
    pto::Shape<1, 1, 1, 128, 128> v38 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v39 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + ((v10 + (unsigned) v34 * (unsigned) v14) + v10 * (unsigned) v13 + v10 * (unsigned) v12), v38, v39);
    TLOAD(v25, v37);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v27, v25);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(v29, v27, v28);
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v40, v29);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  #endif // __DAV_CUBE__

  return;
}

