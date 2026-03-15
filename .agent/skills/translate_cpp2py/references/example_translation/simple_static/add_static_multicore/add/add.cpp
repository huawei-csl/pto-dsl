#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void vec_add_kernel_2d_dynamic(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4, int32_t v5) {
  unsigned v6 = 1024;
  unsigned v7 = 32;
  unsigned v8 = 1;
  unsigned v9 = 0;
  int32_t v10 = 1280;
  int32_t v11 = 32;
  int32_t v12 = 1;
  int64_t v13 = 0;
  int64_t v14 = 4096;
  int64_t v15 = 8192;
  using T = float;
  int64_t v16 = get_block_idx();
  int64_t v17 = get_subblockid();
  int64_t v18 = get_subblockdim();
  int32_t v19 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v16) * (uint64_t) ((int64_t) v18)) + (uint64_t) ((int64_t) v17))) * (uint32_t) v11);
  pto::Shape<1, 1, 1, 32, 32> v20 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v21 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v22 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v1 + (v9 + (unsigned) v19 * (unsigned) v11 + v9 * (unsigned) v12), v20, v21);
  pto::Shape<1, 1, 1, 32, 32> v23 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v24 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v25 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v2 + (v9 + (unsigned) v19 * (unsigned) v11 + v9 * (unsigned) v12), v23, v24);
  pto::Shape<1, 1, 1, 32, 32> v26 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v27 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v28 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v3 + (v9 + (unsigned) v19 * (unsigned) v11 + v9 * (unsigned) v12), v26, v27);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v29 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v29, v13);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v30 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v30, v14);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> v31 = Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>(v4, v5);
  TASSIGN(v31, v15);
  TLOAD(v29, v22);
  TLOAD(v30, v25);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TADD(v31, v29, v30);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v28, v31);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}

