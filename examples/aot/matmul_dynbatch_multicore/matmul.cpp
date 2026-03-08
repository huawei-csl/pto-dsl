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
  int64_t v18 = 16896;
  int64_t v19 = 512;
  int64_t v20 = 0;
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
  Tile<TileType::Mat, float, 128, 32, BLayout::RowMajor, 128, 32> v30;
  TASSIGN(v30, v18);
  Tile<TileType::Mat, float, 128, 32, BLayout::ColMajor, 128, 32, SLayout::RowMajor, 512, PadValue::Null> v31;
  TRESHAPE(v31, v30);
  Tile<TileType::Mat, float, 32, 128, BLayout::RowMajor, 32, 128> v32;
  TASSIGN(v32, v19);
  Tile<TileType::Mat, float, 32, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, 512, PadValue::Null> v33;
  TRESHAPE(v33, v32);
  Tile<TileType::Mat, float, 1, 128, BLayout::RowMajor, 1, 128> v34;
  TASSIGN(v34, v20);
  Tile<TileType::Mat, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v35;
  TRESHAPE(v35, v34);
  Tile<TileType::Left, float, 128, 32, BLayout::RowMajor, 128, 32> v36;
  TASSIGN(v36, v20);
  Tile<TileType::Left, float, 128, 32, BLayout::RowMajor, 128, 32, SLayout::RowMajor, 512, PadValue::Null> v37;
  TRESHAPE(v37, v36);
  Tile<TileType::Right, float, 32, 128, BLayout::RowMajor, 32, 128> v38;
  TASSIGN(v38, v20);
  Tile<TileType::Right, float, 32, 128, BLayout::RowMajor, 32, 128, SLayout::ColMajor, 512, PadValue::Null> v39;
  TRESHAPE(v39, v38);
  Tile<TileType::Acc, float, 128, 128, BLayout::RowMajor, 128, 128> v40;
  TASSIGN(v40, v20);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v41;
  TRESHAPE(v41, v40);
  Tile<TileType::Bias, float, 1, 128, BLayout::RowMajor, 1, 128> v42;
  TASSIGN(v42, v20);
  Tile<TileType::Bias, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v43;
  TRESHAPE(v43, v42);
  for (size_t v44 = (size_t) v28; v44 < ((size_t) ((uint32_t) v29 < (uint32_t) v6 ? v29 : v6)); v44 += v21) {
    int32_t v45 = (int32_t) ((uint32_t) ((int32_t) v44) * (uint32_t) v15);
    for (size_t v46 = (size_t) v13; v46 < ((size_t) v17); v46 += v21) {
      int32_t v47 = (int32_t) v46;
      int32_t v48 = (int32_t) ((uint32_t) v47 * (uint32_t) v16);
      pto::Shape<1, 1, 1, 128, 32> v49 = pto::Shape<1, 1, 1, 128, 32>();
      pto::Stride<16384, 16384, 16384, 128, 1> v50 = pto::Stride<16384, 16384, 16384, 128, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 128, 32>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v51 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 32>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + (v12 + (unsigned) v45 * (unsigned) v15 + (unsigned) v48 * (unsigned) v14), v49, v50);
      pto::Shape<1, 1, 1, 32, 128> v52 = pto::Shape<1, 1, 1, 32, 128>();
      pto::Stride<4096, 4096, 4096, 128, 1> v53 = pto::Stride<4096, 4096, 4096, 128, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND> v54 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 128>, pto::Stride<4096, 4096, 4096, 128, 1>, pto::Layout::ND>(v3 + (v12 + (unsigned) v48 * (unsigned) v15 + v12 * (unsigned) v14), v52, v53);
      pto::Shape<1, 1, 1, 1, 128> v55 = pto::Shape<1, 1, 1, 1, 128>();
      pto::Stride<128, 128, 128, 128, 1> v56 = pto::Stride<128, 128, 128, 128, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v57 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v4 + (v12 + v12 * (unsigned) v15 + v12 * (unsigned) v14), v55, v56);
      TLOAD(v31, v51);
      TLOAD(v33, v54);
      if (v5) {
        TLOAD(v35, v57);
      };
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v37, v31);
      TMOV(v39, v33);
      if (v5) {
        TMOV(v43, v35);
      };
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      if (v47 == v13) {
        if (v5) {
          TMATMUL_BIAS(v41, v37, v39, v43);
        } else {
          TMATMUL(v41, v37, v39);
        };
      } else {
        TMATMUL_ACC(v41, v41, v37, v39);
      };
      set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    };
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    pto::Shape<1, 1, 1, 128, 128> v58 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v59 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v60 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v12 + (unsigned) v45 * (unsigned) v15 + v12 * (unsigned) v14), v58, v59);
    TSTORE(v60, v41);
    set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  }
  #endif // __DAV_CUBE__

  return;
}

