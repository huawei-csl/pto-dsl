#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void RunTMATMULSplitK(__gm__ half *v1, __gm__ half *v2, __gm__ half *v3, __gm__ half *v4, bool v5,
                                        int32_t v6) {
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
  int32_t v20 = (int32_t)((int64_t)v19);
  int32_t v21 = v6 / v20;
  int32_t v22 = v6 % v20 != v11 && v6 < v11 == v20 < v11 ? v21 + v12 : v21;
  int64_t v23 = get_block_idx();
  int32_t v24 = (int32_t)((uint32_t)((int32_t)(int64_t)v23) * (uint32_t)v22);
  int32_t v25 = (int32_t)((uint32_t)v24 + (uint32_t)v22);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> A1_l1;
  TASSIGN(A1_l1, v16);
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> A2_l1;
  TASSIGN(A2_l1, v17);

  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> A1_l0;
  TASSIGN(A1_l0, v18);
  Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> A2_l0;
  TASSIGN(A2_l0, v16);

  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> C1_l0;
  TASSIGN(C1_l0, v18);
  Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> C2_l0;
  TASSIGN(C2_l0, v17);

  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v28;
  TASSIGN(v28, v18);
  Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> B_l0;
  TASSIGN(B_l0, v18);
  pto::Shape<1, 1, 1, 128, 128> v34 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v35 = pto::Stride<16384, 16384, 16384, 128, 1>();

  using GMType =
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>;
  GMType v36 = GMType(v3 + (v10 + v10 * (unsigned)v14 + v10 * (unsigned)v12), v34, v35);
  TLOAD(v28, v36);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(B_l0, v28);

  pto::Shape<1, 1, 1, 128, 128> v39 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v40 = pto::Stride<16384, 16384, 16384, 128, 1>();
  pto::Shape<1, 1, 1, 128, 128> v49 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v50 = pto::Stride<16384, 16384, 16384, 128, 1>();
  pto::Shape<1, 1, 1, 128, 128> v43 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v44 = pto::Stride<16384, 16384, 16384, 128, 1>();
  pto::Shape<1, 1, 1, 128, 128> v46 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v47 = pto::Stride<16384, 16384, 16384, 128, 1>();

  int end = ((size_t)((uint32_t)v25 < (uint32_t)v6 ? v25 : v6));
  // i-2
  int curr = v24 & 1;
  set_flag(PIPE_MTE1, PIPE_MTE2, curr);  // set(1)
  set_flag(PIPE_M, PIPE_MTE1, curr);     // set(3)
  set_flag(PIPE_FIX, PIPE_M, curr);      // set(4)

  // i-1
  // must load the first tile from GM->l1 here since the loop always loads for
  // next iteration
  GMType A_gm_first = GMType(v2 + v24 * v15, v39, v40);
  // this is iteration i-1, in this case -1
  curr = 1 - curr;  // since v24 can start at odd/even i must load the right tile
  TLOAD(curr == 1 ? A1_l1 : A2_l1, A_gm_first);
  set_flag(PIPE_MTE2, PIPE_MTE1, curr);  // set(2) tell MTE1 that MTE2 finished.

  set_flag(PIPE_MTE1, PIPE_MTE2, curr);  // set(1)
  set_flag(PIPE_M, PIPE_MTE1, curr);     // set(3)
  set_flag(PIPE_FIX, PIPE_M, curr);      // set(4)

  for (size_t i = v24; i < end; i += 1) {
    curr = i & 1;
    // Global memory for A tiles
    GMType v45 = GMType(v2 + (i + 1) * v15, v43, v44);
    // GM tile C_1 and C_2
    GMType v48 = GMType(v1 + i * v15, v46, v47);

    // Start loading the tile used in matmul at iteration i+1
    wait_flag(PIPE_MTE1, PIPE_MTE2, curr);  // (1, i-2) wait until the MOV at i-2 has completed
    if (i + 1 < end) {
      TLOAD(curr == 1 ? A1_l1 : A2_l1, v45);
      set_flag(PIPE_MTE2, PIPE_MTE1, curr);  // set(2, i+1) notify the mov below in iteration i+1 that the load completed
    }

    // mov
    wait_flag(PIPE_MTE2, PIPE_MTE1, 1 - curr); // (2, i-1) last iteration loaded the tile into l1, so
                                         // for us to move to l0 we wait for last it
    wait_flag(PIPE_M, PIPE_MTE1, curr);  // (3, i-2) make sure the matmul from
                                         // i-2 finished so we can overwrite l0A
    TMOV(curr == 0 ? A1_l0 : A2_l0, curr == 0 ? A1_l1 : A2_l1);
    if (i + 2 < end) {
      set_flag(PIPE_MTE1, PIPE_MTE2, curr);  // set(1, i+2) notify load at iteration i+2 that it's ready
    }
    set_flag(PIPE_MTE1, PIPE_M, curr);  // set(5, i) simply notify matmul at it. i it is ready.

    // matmul
    wait_flag(PIPE_FIX, PIPE_M, curr);  // (4, i-2) wait until the STORE at it.
                                        // i-2 has written back from L0C
    wait_flag(PIPE_MTE1, PIPE_M, curr);  // (5, i) need the tile that is moved into L0A at iteration i
    TMATMUL(curr == 0 ? C1_l0 : C2_l0, curr == 0 ? A1_l0 : A2_l0, B_l0);
    set_flag(PIPE_M, PIPE_FIX, curr);  // set(6, i) notify store in this
                                       // iteration i, that matmul is done
    if (i + 2 < end) {
      set_flag(PIPE_M, PIPE_MTE1, curr);  // set(3, i+2) notify mov in iteration i+2, that matmul is done
    }

    // store
    wait_flag(PIPE_M, PIPE_FIX, curr);  // (6, i) wait for matmul in it. i to be done
    TSTORE(v48, curr == 0 ? C1_l0 : C2_l0);
    if (i + 2 < end) {
      set_flag(PIPE_FIX, PIPE_M, curr);  // set(4, i+2) notify matmul in i+2 that store is complete
    }
  }

#endif  // __DAV_CUBE__

  return;
}
