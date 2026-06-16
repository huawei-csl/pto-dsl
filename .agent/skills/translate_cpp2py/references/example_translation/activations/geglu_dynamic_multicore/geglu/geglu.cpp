#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

__global__ AICORE void _kernel(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, int32_t v4, int32_t v5) {
  unsigned v6 = 1;
  unsigned v7 = 0;
  const int32_t v8 = 16384;
  const int32_t v9 = 1;
  const int32_t v10 = 0;
  const int64_t v11 = 0;
  const int64_t v12 = 32768;
  const int64_t v13 = 65536;
  const int64_t v14 = 98304;
  const int64_t v15 = 131072;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  if (v5 > v10) {
    if (v5 <= v8) {
      int64_t v16 = get_block_idx();
      int64_t v17 = get_subblockid();
      int64_t v18 = get_subblockdim();
      int64_t v19 = (int64_t) v18;
      int64_t v20 = get_block_num();
      int32_t v21 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v20) * (uint64_t) v19);
      int32_t v22 = v4 / v21;
      int32_t v23 = v4 % v21 != v10 && v4 < v10 == v21 < v10 ? v22 + v9 : v22;
      int32_t v24 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v16) * (uint64_t) v19) + (uint64_t) ((int64_t) v17))) * (uint32_t) v23);
      int32_t v25 = (int32_t) ((uint32_t) v24 + (uint32_t) v23);
      int32_t v26 = (int32_t) ((uint32_t) ((uint32_t) v25 < (uint32_t) v4 ? v25 : v4) - (uint32_t) v24);
      int32_t v27 = (int32_t) ((uint32_t) v4 * (uint32_t) v5);
      if (v26 > v10) {
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v28;
        TASSIGN(v28, v11);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v29 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v5);
        __ubuf__ half* v30 = v28.data();
        uint64_t v31 = reinterpret_cast<uint64_t>(v30);
        TASSIGN(v29, v31);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32;
        TASSIGN(v32, v12);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v5);
        __ubuf__ half* v34 = v32.data();
        uint64_t v35 = reinterpret_cast<uint64_t>(v34);
        TASSIGN(v33, v35);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v36;
        TASSIGN(v36, v13);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v5);
        __ubuf__ half* v38 = v36.data();
        uint64_t v39 = reinterpret_cast<uint64_t>(v38);
        TASSIGN(v37, v39);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v40;
        TASSIGN(v40, v14);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v5);
        __ubuf__ half* v42 = v40.data();
        uint64_t v43 = reinterpret_cast<uint64_t>(v42);
        TASSIGN(v41, v43);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v44;
        TASSIGN(v44, v15);
        Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v5);
        __ubuf__ half* v46 = v44.data();
        uint64_t v47 = reinterpret_cast<uint64_t>(v46);
        TASSIGN(v45, v47);
        for (size_t v48 = (size_t) v10; v48 < ((size_t) v26); v48 += (size_t) v9) {
          int32_t v49 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v24 + (uint32_t) ((int32_t) v48)) * (uint32_t) v5);
          unsigned v50 = (unsigned) v5 * v6;
          pto::Shape<1, 1, 1, 1, -1> v51 = pto::Shape<1, 1, 1, 1, -1>(v5);
          pto::Stride<-1, -1, -1, -1, 1> v52 = pto::Stride<-1, -1, -1, -1, 1>(v50, v50, v50, v50);
          GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v53 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) v49 * (unsigned) v9), v51, v52);
          unsigned v54 = (unsigned) v5 * v6;
          pto::Shape<1, 1, 1, 1, -1> v55 = pto::Shape<1, 1, 1, 1, -1>(v5);
          pto::Stride<-1, -1, -1, -1, 1> v56 = pto::Stride<-1, -1, -1, -1, 1>(v54, v54, v54, v54);
          GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v57 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v2 + (v7 + (unsigned) v49 * (unsigned) v9), v55, v56);
          unsigned v58 = (unsigned) v5 * v6;
          pto::Shape<1, 1, 1, 1, -1> v59 = pto::Shape<1, 1, 1, 1, -1>(v5);
          pto::Stride<-1, -1, -1, -1, 1> v60 = pto::Stride<-1, -1, -1, -1, 1>(v58, v58, v58, v58);
          GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v61 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v3 + (v7 + (unsigned) v49 * (unsigned) v9), v59, v60);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TLOAD(v29, v53);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          TLOAD(v33, v57);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          TSUB(v45, v29, v29);
          pipe_barrier(PIPE_V);
          TEXP(v37, v45);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          TADD(v41, v29, v29);
          pipe_barrier(PIPE_V);
          TEXP(v41, v41);
          pipe_barrier(PIPE_V);
          TSUB(v45, v41, v37);
          pipe_barrier(PIPE_V);
          TADD(v41, v41, v37);
          pipe_barrier(PIPE_V);
          TDIV(v45, v45, v41);
          pipe_barrier(PIPE_V);
          TADD(v41, v37, v45);
          pipe_barrier(PIPE_V);
          TMUL(v41, v29, v41);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          TADD(v45, v37, v37);
          pipe_barrier(PIPE_V);
          TDIV(v41, v41, v45);
          pipe_barrier(PIPE_V);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
          TMUL(v41, v41, v33);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          pipe_barrier(PIPE_MTE3);
          TSTORE(v61, v41);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        };
      };
    };
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
