#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void RunTMATMULSplitK(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ half* v4, bool v5, int32_t v6) {
  unsigned v7 = 16384;
  unsigned v8 = 128;
  unsigned v9 = 1;
  unsigned v10 = 0;
  int32_t v11 = 0;
  int32_t v12 = 1;
  int32_t v13 = 2;
  int32_t v14 = 128;
  int32_t v15 = 16384;
  int64_t v16 = 32768;
  int64_t v17 = 65536;
  int64_t v18 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  int64_t v19 = get_block_num();
  int32_t v20 = (int32_t) ((int64_t) v19);
  int32_t v21 = v6 / v20;
  int32_t v22 = v6 % v20 != v11 && v6 < v11 == v20 < v11 ? v21 + v12 : v21;
  int64_t v23 = get_block_idx();
  int32_t v24 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v23) * (uint32_t) v22);
  int32_t v25 = (int32_t) ((uint32_t) v24 + (uint32_t) v22);
  int32_t v26 = (uint32_t) v25 < (uint32_t) v6 ? v25 : v6;
  Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, 128, 128> v27;
  TASSIGN(v27, v16);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v28;
  TRESHAPE(v28, v27);
  Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, 128, 128> v29;
  TASSIGN(v29, v17);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v30;
  TRESHAPE(v30, v29);
  Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, 128, 128> v31;
  TASSIGN(v31, v18);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v32;
  TRESHAPE(v32, v31);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128> v33;
  TASSIGN(v33, v18);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v34;
  TRESHAPE(v34, v33);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128> v35;
  TASSIGN(v35, v16);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v36;
  TRESHAPE(v36, v35);
  Tile<TileType::Acc, float, 128, 128, BLayout::RowMajor, 128, 128> v37;
  TASSIGN(v37, v18);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v38;
  TRESHAPE(v38, v37);
  Tile<TileType::Acc, float, 128, 128, BLayout::RowMajor, 128, 128> v39;
  TASSIGN(v39, v17);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v40;
  TRESHAPE(v40, v39);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128> v41;
  TASSIGN(v41, v18);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v42;
  TRESHAPE(v42, v41);
  pto::Shape<1, 1, 1, 128, 128> v43 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v44 = pto::Stride<16384, 16384, 16384, 128, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v45 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v3 + (v10 + v10 * (unsigned) v14 + v10 * (unsigned) v12), v43, v44);
  TLOAD(v32, v45);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v42, v32);
  pto::Shape<1, 1, 1, 128, 128> v46 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v47 = pto::Stride<16384, 16384, 16384, 128, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v48 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + ((v10 + (unsigned) v24 * (unsigned) v15) + v10 * (unsigned) v14 + v10 * (unsigned) v12), v46, v47);
  int32_t v49 = (int32_t) ((uint32_t) v12 - (uint32_t) (v24 % v13));
  if (v49 == v12) {
    TLOAD(v28, v48);
  } else {
    TLOAD(v30, v48);
  }
  if (v49 == v11) {
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  } else {
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  }
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  for (size_t v50 = (size_t) v24; v50 < ((size_t) v26); v50 += (size_t) v12) {
    int32_t v51 = (int32_t) v50;
    int32_t v52 = v51 % v13;
    int32_t v53 = (int32_t) ((uint32_t) v51 + (uint32_t) v12);
    pto::Shape<1, 1, 1, 128, 128> v54 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v55 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v56 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v2 + ((v10 + (unsigned) v53 * (unsigned) v15) + v10 * (unsigned) v14 + v10 * (unsigned) v12), v54, v55);
    pto::Shape<1, 1, 1, 128, 128> v57 = pto::Shape<1, 1, 1, 128, 128>();
    pto::Stride<16384, 16384, 16384, 128, 1> v58 = pto::Stride<16384, 16384, 16384, 128, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v59 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + ((v10 + (unsigned) v51 * (unsigned) v15) + v10 * (unsigned) v14 + v10 * (unsigned) v12), v57, v58);
    bool v60 = v52 == v11;
    if (v60) {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    } else {
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    };
    if (v53 < v26) {
      if (v52 == v12) {
        TLOAD(v28, v56);
      } else {
        TLOAD(v30, v56);
      };
      if (v60) {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      };
    };
    if ((int32_t) ((uint32_t) v12 - (uint32_t) v52) == v11) {
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    } else {
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    };
    if (v60) {
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    } else {
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    };
    if (v60) {
      TMOV(v34, v28);
    } else {
      TMOV(v36, v30);
    };
    bool v61 = (int32_t) ((uint32_t) v51 + (uint32_t) v13) < v26;
    if (v61) {
      if (v60) {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      } else {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
      };
    };
    if (v60) {
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    } else {
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    };
    if (v60) {
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    } else {
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    };
    if (v60) {
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    } else {
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    };
    if (v60) {
      TMATMUL(v38, v34, v42);
    } else {
      TMATMUL(v40, v36, v42);
    };
    if (v60) {
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    } else {
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    };
    if (v61) {
      if (v60) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      } else {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
      };
    };
    if (v60) {
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    } else {
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    };
    if (v60) {
      TSTORE(v59, v38);
    } else {
      TSTORE(v59, v40);
    };
    if (v61) {
      if (v60) {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      } else {
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      };
    };
    pipe_barrier(PIPE_MTE2);
  }
  #endif // __DAV_CUBE__

  return;
}

