#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void RunTMATMULSplitK(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, bool v5, int32_t v6) {
  unsigned v7 = 4096;
  unsigned v8 = 16384;
  unsigned v9 = 32;
  unsigned v10 = 128;
  unsigned v11 = 1;
  unsigned v12 = 0;
  int32_t v13 = 0;
  int32_t v14 = 1;
  int32_t v15 = 128;
  int32_t v16 = 32;
  int32_t v17 = 4;
  int64_t v18 = 16384;
  int64_t v19 = 0;
  int64_t v20 = 32768;
  using T = float;
  size_t v21 = (size_t) v14;

  #if defined(__DAV_CUBE__)
  int32_t v22 = (int32_t) ((uint32_t) v6 * (uint32_t) v15);
  int64_t v23 = get_block_num();
  int32_t v24 = (int32_t) ((int64_t) v23);
  int32_t v25 = v6 / v24;
  int32_t v26 = v6 % v24 != v13 && v6 < v13 == v24 < v13 ? v25 + v14 : v25;
  int64_t v27 = get_block_idx();
  int32_t v28 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v27) * (uint32_t) v26);
  int32_t v29 = (int32_t) ((uint32_t) v28 + (uint32_t) v26);
  Tile<TileType::Mat, float, 128, 32, BLayout::ColMajor, 128, 32, SLayout::RowMajor, 512, PadValue::Null> v30;
  TASSIGN(v30, v18);
  Tile<TileType::Mat, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v31;
  TASSIGN(v31, v19);
  Tile<TileType::Mat, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v32;
  TASSIGN(v32, v20);
  Tile<TileType::Left, float, 128, 32, BLayout::RowMajor, 128, 32, SLayout::RowMajor, 512, PadValue::Null> v33;
  TASSIGN(v33, v19);
  Tile<TileType::Right, float, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v34;
  TASSIGN(v34, v19);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v35;
  TASSIGN(v35, v19);
  Tile<TileType::Bias, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v36;
  TASSIGN(v36, v19);
  for (size_t v37 = (size_t) v28; v37 < ((size_t) ((uint32_t) v29 < (uint32_t) v6 ? v29 : v6)); v37 += v21) {
    int32_t v38 = (int32_t) ((uint32_t) ((int32_t) v37) * (uint32_t) v15);
    for (size_t v39 = (size_t) v13; v39 < ((size_t) v17); v39 += v21) {
      int32_t v40 = (int32_t) v39;
      int32_t v41 = (int32_t) ((uint32_t) v40 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 128, 32> v42 = pto::Shape<1, 1, 1, 128, 32>();
      pto::Stride<16384, 16384, 16384, 128, 1> v43 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 128, 32>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 32>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v38 * (unsigned) v15 + (unsigned) v41 * (unsigned) v14), v42, v43);
      pto::Shape<1, 1, 1, 32, 128> v45 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<4096, 4096, 4096, 128, 1> v46 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v47 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v41 * (unsigned) v15 + v12 * (unsigned) v14), v45, v46);
      pto::Shape<1, 1, 1, 1, 128> v48 = pto::Shape<1, 1, 1, 1, 128>();
      pto::Stride<128, 128, 128, 128, 1> v49 = pto::Stride<128, 128, 128, 128, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v50 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v4 + (v12 + v12 * (unsigned) v15 + v12 * (unsigned) v14), v48, v49);
      TLOAD(v30, v44);
      TLOAD(v31, v47);
      if (v5) {
        TLOAD(v32, v50);
      };
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v33, v30);
      TMOV(v34, v31);
      if (v5) {
        TMOV(v36, v32);
      };
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v40 == v13) {
        if (v5) {
          TMATMUL_BIAS(v35, v33, v34, v36);
        } else {
          TMATMUL(v35, v33, v34);
        };
      } else {
        TMATMUL_ACC(v35, v35, v33, v34);
      };
      set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 128, 128> v51 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v52 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v38 * (unsigned) v15 + v12 * (unsigned) v14), v51, v52);
    TSTORE(v53, v35);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  #endif // __DAV_CUBE__

  return;
}

