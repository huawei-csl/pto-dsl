#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void RunTMATMULSplitK(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, bool v5) {
  unsigned v6 = 1024;
  unsigned v7 = 8192;
  unsigned v8 = 256;
  unsigned v9 = 32;
  unsigned v10 = 1;
  unsigned v11 = 0;
  int32_t v12 = 0;
  int32_t v13 = 1;
  int32_t v14 = 32;
  int32_t v15 = 256;
  int32_t v16 = 8;
  int64_t v17 = 0;
  int64_t v18 = 4096;
  int64_t v19 = 8192;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v20;
  TASSIGN(v20, v17);
  Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v21;
  TASSIGN(v21, v18);
  Tile<TileType::Mat, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v19);
  Tile<TileType::Left, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v23;
  TASSIGN(v23, v17);
  Tile<TileType::Right, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::ColMajor, 512, PadValue::Null> v24;
  TASSIGN(v24, v17);
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v25;
  TASSIGN(v25, v17);
  Tile<TileType::Bias, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v17);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  for (size_t v27 = (size_t) v12; v27 < ((size_t) v16); v27 += (size_t) v13) {
    int32_t v28 = (int32_t) v27;
    int32_t v29 = (int32_t) ((uint32_t) v28 * (uint32_t) v14);
    pto::Shape<1, 1, 1, 32, 32> v30 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<8192, 8192, 8192, 256, 1> v31 = pto::Stride<8192, 8192, 8192, 256, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v32 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v2 + (v11 + v11 * (unsigned) v15 + (unsigned) v29 * (unsigned) v13), v30, v31);
    pto::Shape<1, 1, 1, 32, 32> v33 = pto::Shape<1, 1, 1, 32, 32>();
    pto::Stride<1024, 1024, 1024, 32, 1> v34 = pto::Stride<1024, 1024, 1024, 32, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v35 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v3 + (v11 + (unsigned) v29 * (unsigned) v14 + v11 * (unsigned) v13), v33, v34);
    pto::Shape<1, 1, 1, 1, 32> v36 = pto::Shape<1, 1, 1, 1, 32>();
    pto::Stride<32, 32, 32, 32, 1> v37 = pto::Stride<32, 32, 32, 32, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND> v38 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 32>, pto::Stride<32, 32, 32, 32, 1>, pto::Layout::ND>(v4 + (v11 + v11 * (unsigned) v14 + v11 * (unsigned) v13), v36, v37);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v20, v32);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    TLOAD(v21, v35);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    if (v5) {
      pipe_barrier(PIPE_MTE2);
      TLOAD(v22, v38);
    };
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    pipe_barrier(PIPE_MTE1);
    TMOV(v23, v20);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(v24, v21);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    if (v5) {
      TMOV(v26, v22);
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v28 == v12) {
      if (v5) {
        TMATMUL_BIAS(v25, v23, v24, v26);
      } else {
        TMATMUL(v25, v23, v24);
      };
    } else {
      TMATMUL_ACC(v25, v25, v23, v24);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 32> v39 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v40 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v41 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v1 + (v11 + v11 * (unsigned) v14 + v11 * (unsigned) v13), v39, v40);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v41, v25);
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

