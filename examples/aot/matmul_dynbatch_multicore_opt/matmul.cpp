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
  Tile<TileType::Mat, float, 128, 128, BLayout::RowMajor, 128, 128> v25;
  TASSIGN(v25, v15);
  Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v26;
  TRESHAPE(v26, v25);
  Tile<TileType::Mat, float, 128, 128, BLayout::RowMajor, 128, 128> v27;
  TASSIGN(v27, v16);
  Tile<TileType::Mat, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v28;
  TRESHAPE(v28, v27);
  Tile<TileType::Left, float, 128, 128, BLayout::RowMajor, 128, 128> v29;
  TASSIGN(v29, v16);
  Tile<TileType::Left, float, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v30;
  TRESHAPE(v30, v29);
  Tile<TileType::Right, float, 128, 128, BLayout::RowMajor, 128, 128> v31;
  TASSIGN(v31, v16);
  Tile<TileType::Right, float, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v32;
  TRESHAPE(v32, v31);
  Tile<TileType::Acc, float, 128, 128, BLayout::RowMajor, 128, 128> v33;
  TASSIGN(v33, v16);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v34;
  TRESHAPE(v34, v33);
  pto::Shape<1, 1, 1, 128, 128> v35 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v36 = pto::Stride<16384, 16384, 16384, 128, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v37 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v10 + v10 * (unsigned) v13 + v10 * (unsigned) v12), v35, v36);
  TLOAD(v28, v37);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v32, v28);
  for (size_t v38 = (size_t) v23; v38 < ((size_t) ((uint32_t) v24 < (uint32_t) v6 ? v24 : v6)); v38 += (size_t) v12) {
    int32_t v39 = (int32_t) v38;
    pto::Shape<1, 1, 1, 128, 128> v40 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v41 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v42 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + ((v10 + (unsigned) v39 * (unsigned) v14) + v10 * (unsigned) v13 + v10 * (unsigned) v12), v40, v41);
    pto::Shape<1, 1, 1, 128, 128> v43 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v44 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v45 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + ((v10 + (unsigned) v39 * (unsigned) v14) + v10 * (unsigned) v13 + v10 * (unsigned) v12), v43, v44);
    TLOAD(v26, v42);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v30, v26);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    TMATMUL(v34, v30, v32);
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(v45, v34);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  #endif // __DAV_CUBE__

  return;
}

