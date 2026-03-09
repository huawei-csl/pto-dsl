#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void kernel(__gm__ half* v1) {
  unsigned v2 = 16384;
  unsigned v3 = 128;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 1;
  int32_t v7 = 128;
  int64_t v8 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v9;
  TASSIGN(v9, v8);
  pto::Shape<1, 1, 1, 128, 128> v10 = pto::Shape<1, 1, 1, 128, 128>();
  pto::Stride<16384, 16384, 16384, 128, 1> v11 = pto::Stride<16384, 16384, 16384, 128, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND> v12 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<16384, 16384, 16384, 128, 1>, pto::Layout::ND>(v1 + (v5 + v5 * (unsigned) v7 + v5 * (unsigned) v6), v10, v11);
  TLOAD(v9, v12);
  #endif // __DAV_CUBE__

  return;
}

