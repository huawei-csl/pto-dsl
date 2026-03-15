#define MEMORY_BASE
#include <pto/pto-inst.hpp>

using namespace pto;

extern "C" __global__ AICORE void repro_manual_acc_to_mat(
    __gm__ float *out_ptr, __gm__ half *in_ptr, int32_t n_i32) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  const uint32_t n = static_cast<uint32_t>(n_i32);

  using TensorShapeIn = TileShape2D<half, 128, 128, Layout::ND>;
  using TensorStridesIn = BaseShape2D<half, 128, 128, Layout::ND>;
  using TensorShapeOut = TileShape2D<float, 128, 128, Layout::ND>;
  using TensorStridesOut = BaseShape2D<float, 128, 128, Layout::ND>;

  using GlobalTensorIn =
      GlobalTensor<half, TensorShapeIn, TensorStridesIn, Layout::ND>;
  using GlobalTensorOut =
      GlobalTensor<float, TensorShapeOut, TensorStridesOut, Layout::ND>;

  // MAT tile is half.
  using TileL1 = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128,
                      128, SLayout::RowMajor, 512>;
  using TileL0A = TileLeft<half, 128, 128>;
  using TileL0B = TileRight<half, 128, 128>;
  // ACC tile is float.
  using TileL0C = TileAcc<float, 128, 128>;

  GlobalTensorIn in_gm(in_ptr);
  GlobalTensorOut out_gm(out_ptr);

  TileL1 y_l1;
  TileL0A a_l0;
  TileL0B b_l0;
  TileL0C c_l0;

  TASSIGN(y_l1, 0x0);
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);
  TASSIGN(c_l0, 0x0);

  y_l1.SetValidRow(n);
  y_l1.SetValidCol(n);
  a_l0.SetValidRow(n);
  a_l0.SetValidCol(n);
  b_l0.SetValidRow(n);
  b_l0.SetValidCol(n);
  c_l0.SetValidRow(n);
  c_l0.SetValidCol(n);

  TLOAD(y_l1, in_gm);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

  TMOV(a_l0, y_l1);
  TMOV(b_l0, y_l1);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  TMATMUL(c_l0, a_l0, b_l0);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

  TMOV(y_l1, c_l0);  // ACC(float, TileL0C) -> MAT(half, TileL1): known manual working path
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  TMATMUL(c_l0, a_l0, b_l0);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(out_gm, c_l0);
#endif
}
