#include "pto/pto-inst.hpp"
using namespace pto;

template <typename To, typename From>
static inline To ptoas_bitcast(From from) {
  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
  To to;
  __builtin_memcpy(&to, &from, sizeof(To));
  return to;
}

__global__ AICORE void sync_kernel_dyn(__gm__ float* v1, __gm__ float* v2, int32_t v3) {
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 0;
  int32_t v7 = 1;
  int32_t v8 = 10;
  int32_t v9 = 32;
  int64_t v10 = 0;
  int64_t v11 = 128;
  using T = float;

  #if defined(__DAV_VEC__)
  uint32_t v12 = (uint32_t) v3;
  uint32_t v13 = (uint32_t) v8;
  uint32_t v14 = v12 / v13;
  int32_t v15 = (int32_t) v14;
  uint32_t v16 = (uint32_t) v15;
  uint32_t v17 = (uint32_t) v9;
  uint32_t v18 = v16 / v17;
  int32_t v19 = (int32_t) v18;
  int64_t v20 = get_block_idx();
  int32_t v21 = (int32_t) v20;
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v10);
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v11);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  for (int32_t v24 = v6; v24 < v19; v24 += v7) {
    uint32_t v25 = (uint32_t) v21;
    uint32_t v26 = (uint32_t) v15;
    uint32_t v27 = v25 * v26;
    int32_t v28 = (int32_t) v27;
    uint32_t v29 = (uint32_t) v24;
    uint32_t v30 = (uint32_t) v9;
    uint32_t v31 = v29 * v30;
    int32_t v32 = (int32_t) v31;
    uint32_t v33 = (uint32_t) v28;
    uint32_t v34 = (uint32_t) v32;
    uint32_t v35 = v33 + v34;
    int32_t v36 = (int32_t) v35;
    unsigned v37 = (unsigned) v36;
    unsigned v38 = (unsigned) v7;
    unsigned v39 = v37 * v38;
    unsigned v40 = v5 + v39;
    __gm__ float* v41 = v1 + v40;
    using GTShape_505226096 = pto::Shape<1, 1, 1, 1, 32>;;
    using GTStride_505226096 = pto::Stride<32, 32, 32, 32, 1>;;
    constexpr pto::Layout GT_505226096_layout = pto::Layout::ND;;
    GTShape_505226096 v42 = GTShape_505226096();
    GTStride_505226096 v43 = GTStride_505226096();
    using GT_505226096 = GlobalTensor<float, GTShape_505226096, GTStride_505226096, GT_505226096_layout>;;
    GT_505226096 v44 = GT_505226096(v41, v42, v43);
    unsigned v45 = (unsigned) v36;
    unsigned v46 = (unsigned) v7;
    unsigned v47 = v45 * v46;
    unsigned v48 = v5 + v47;
    __gm__ float* v49 = v2 + v48;
    using GTShape_505044128 = pto::Shape<1, 1, 1, 1, 32>;;
    using GTStride_505044128 = pto::Stride<32, 32, 32, 32, 1>;;
    constexpr pto::Layout GT_505044128_layout = pto::Layout::ND;;
    GTShape_505044128 v50 = GTShape_505044128();
    GTStride_505044128 v51 = GTStride_505044128();
    using GT_505044128 = GlobalTensor<float, GTShape_505044128, GTStride_505044128, GT_505044128_layout>;;
    GT_505044128 v52 = GT_505044128(v49, v50, v51);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v22, v44);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TRELU(v23, v22);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v52, v23);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

extern "C" void call_kernel( uint32_t blockDim, void* stream, void* v1, void* v2, uint32_t n) {
    sync_kernel_dyn<<<blockDim, nullptr, stream>>>(( __gm__ float *)v1, (__gm__ float *)v2, n);
}
