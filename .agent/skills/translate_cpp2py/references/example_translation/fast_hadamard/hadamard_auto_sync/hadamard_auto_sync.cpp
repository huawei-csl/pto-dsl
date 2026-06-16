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

__global__ AICORE void fast_hadamard_autosync(__gm__ half* v1, int32_t v2, int32_t v3, int32_t v4) {
  unsigned v5 = 16384;
  unsigned v6 = 1;
  unsigned v7 = 0;
  const int32_t v8 = 2;
  const int32_t v9 = 1;
  const int32_t v10 = 0;
  const int32_t v11 = 8192;
  const int64_t v12 = 0;
  const int64_t v13 = 32768;
  const int64_t v14 = 49152;
  const int64_t v15 = 65536;
  const int64_t v16 = 98304;
  const int64_t v17 = 114688;
  using T = float;
  size_t v18 = (size_t) v10;
  size_t v19 = (size_t) v9;
  size_t v20 = (size_t) v4;
  int64_t v21 = get_block_idx();
  int64_t v22 = get_subblockid();
  int64_t v23 = get_subblockdim();
  int64_t v24 = (int64_t) v23;
  int64_t v25 = get_block_num();
  int32_t v26 = (int32_t) ((int64_t) (uint64_t) ((int64_t) v25) * (uint64_t) v24);

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int32_t v27 = v2 / v26;
  int32_t v28 = v2 % v26 != v10 && v2 < v10 == v26 < v10 ? v27 + v9 : v27;
  int32_t v29 = (int32_t) ((uint32_t) ((int32_t) (int64_t) ((uint64_t) ((int64_t) (uint64_t) ((int64_t) v21) * (uint64_t) v24) + (uint64_t) ((int64_t) v22))) * (uint32_t) v28);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
  if (v29 < v2) {
    int32_t v30 = (int32_t) ((uint32_t) v29 + (uint32_t) v28) > v2 ? (int32_t) ((uint32_t) v2 - (uint32_t) v29) : v28;
    if (v30 > v10) {
      int32_t v31 = (int32_t) ((uint32_t) v2 * (uint32_t) v3);
      Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v32;
      TASSIGN(v32, v12);
      Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v33 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v3);
      __ubuf__ half* v34 = v32.data();
      uint64_t v35 = reinterpret_cast<uint64_t>(v34);
      TASSIGN(v33, v35);
      int32_t v36 = v3 / v8;
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v37;
      TASSIGN(v37, v13);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v38 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36);
      __ubuf__ half* v39 = v37.data();
      uint64_t v40 = reinterpret_cast<uint64_t>(v39);
      TASSIGN(v38, v40);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v41;
      TASSIGN(v41, v14);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v42 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36);
      __ubuf__ half* v43 = v41.data();
      uint64_t v44 = reinterpret_cast<uint64_t>(v43);
      TASSIGN(v42, v44);
      Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, 16384, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v45;
      TASSIGN(v45, v15);
      Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v46 = Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v3);
      __ubuf__ half* v47 = v45.data();
      uint64_t v48 = reinterpret_cast<uint64_t>(v47);
      TASSIGN(v46, v48);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v49;
      TASSIGN(v49, v16);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v50 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36);
      __ubuf__ half* v51 = v49.data();
      uint64_t v52 = reinterpret_cast<uint64_t>(v51);
      TASSIGN(v50, v52);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, 8192, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v53;
      TASSIGN(v53, v17);
      Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v54 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v36);
      __ubuf__ half* v55 = v53.data();
      uint64_t v56 = reinterpret_cast<uint64_t>(v55);
      TASSIGN(v54, v56);
      for (size_t v57 = v18; v57 < ((size_t) v30); v57 += v19) {
        int32_t v58 = (int32_t) v57;
        int32_t v59 = (int32_t) ((uint32_t) v30 - (uint32_t) v58);
        int32_t v60 = v59 < v9 ? v59 : v9;
        size_t v61 = (size_t) v60;
        if (v60 > v10) {
          int32_t v62 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v29 + (uint32_t) v58) * (uint32_t) v3);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
          if (v58 % v8 == v10) {
            for (size_t v63 = v18; v63 < v61; v63 += v19) {
              unsigned v64 = (unsigned) v3 * v6;
              pto::Shape<1, 1, 1, 1, -1> v65 = pto::Shape<1, 1, 1, 1, -1>(v3);
              pto::Stride<-1, -1, -1, -1, 1> v66 = pto::Stride<-1, -1, -1, -1, 1>(v64, v64, v64, v64);
              GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v67 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v62 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v63) * (uint32_t) v3)) * (unsigned) v9), v65, v66);
              __ubuf__ half* v68 = v33.data();
              int64_t v69 = (int64_t) v3;
              int32_t v70 = (int32_t) ((int64_t) (uint64_t) v69 - (uint64_t) ((int64_t) (uint64_t) v12 % (uint64_t) v69));
              Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v71 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v70 < v11 ? v70 : v11);
              uint64_t v72 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v68 + (v7 + v7 * v5) + v7 * v6));
              TASSIGN(v71, v72);
              __ubuf__ half* v73 = v33.data();
              int32_t v74 = (int32_t) ((int64_t) (uint64_t) v69 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v36) % (uint64_t) v69));
              Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v75 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v74 < v11 ? v74 : v11);
              uint64_t v76 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v73 + (v7 + v7 * v5) + (unsigned) v36 * v6));
              TASSIGN(v75, v76);
              wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
              TLOAD(v33, v67);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              for (size_t v77 = v18; v77 < v20; v77 += v19) {
                pipe_barrier(PIPE_V);
                TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, MaskPattern::P0101>(v38, v33);
                TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, MaskPattern::P1010>(v42, v33);
                pipe_barrier(PIPE_V);
                TADD(v71, v38, v42);
                pipe_barrier(PIPE_V);
                TSUB(v75, v38, v42);
              };
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              pipe_barrier(PIPE_MTE3);
              TSTORE(v67, v33);
              set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
            };
          } else {
            for (size_t v78 = v18; v78 < v61; v78 += v19) {
              unsigned v79 = (unsigned) v3 * v6;
              pto::Shape<1, 1, 1, 1, -1> v80 = pto::Shape<1, 1, 1, 1, -1>(v3);
              pto::Stride<-1, -1, -1, -1, 1> v81 = pto::Stride<-1, -1, -1, -1, 1>(v79, v79, v79, v79);
              GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v82 = GlobalTensor<half, pto::Shape<1, 1, 1, 1, -1>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v7 + (unsigned) ((int32_t) (uint32_t) v62 + (uint32_t) ((int32_t) (uint32_t) ((int32_t) v78) * (uint32_t) v3)) * (unsigned) v9), v80, v81);
              __ubuf__ half* v83 = v46.data();
              int64_t v84 = (int64_t) v3;
              int32_t v85 = (int32_t) ((int64_t) (uint64_t) v84 - (uint64_t) ((int64_t) (uint64_t) v12 % (uint64_t) v84));
              Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v86 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v85 < v11 ? v85 : v11);
              uint64_t v87 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v83 + (v7 + v7 * v5) + v7 * v6));
              TASSIGN(v86, v87);
              __ubuf__ half* v88 = v46.data();
              int32_t v89 = (int32_t) ((int64_t) (uint64_t) v84 - (uint64_t) ((int64_t) (uint64_t) ((int64_t) v36) % (uint64_t) v84));
              Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null> v90 = Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>(v89 < v11 ? v89 : v11);
              uint64_t v91 = reinterpret_cast<uint64_t>((__ubuf__ half*) (v88 + (v7 + v7 * v5) + (unsigned) v36 * v6));
              TASSIGN(v90, v91);
              wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
              TLOAD(v46, v82);
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
              for (size_t v92 = v18; v92 < v20; v92 += v19) {
                pipe_barrier(PIPE_V);
                TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, MaskPattern::P0101>(v50, v46);
                TGATHER<Tile<TileType::Vec, half, 1, 8192, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, Tile<TileType::Vec, half, 1, 16384, BLayout::RowMajor, 1, -1, SLayout::NoneBox, 512, PadValue::Null, CompactMode::Null>, MaskPattern::P1010>(v54, v46);
                pipe_barrier(PIPE_V);
                TADD(v86, v50, v54);
                pipe_barrier(PIPE_V);
                TSUB(v90, v50, v54);
              };
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
              pipe_barrier(PIPE_MTE3);
              TSTORE(v82, v46);
              set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
            };
          };
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
          set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        };
      };
    };
  }
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
  #endif // __DAV_VEC__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
