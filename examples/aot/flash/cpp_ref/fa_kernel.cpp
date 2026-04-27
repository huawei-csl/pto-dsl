/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>

#include <pto/pto-inst.hpp>
#include "runtime/rt.h"
#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__)
#include <pto/npu/a2a3/custom/TSyncCVID.hpp>
#include <pto/npu/a2a3/custom/TSync_Custom.hpp>
#define UF_ENABLE 1
#elif defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include <pto/npu/a5/custom/TSyncCVID.hpp>
#include <pto/npu/a5/custom/TSync_Custom.hpp>
#define UF_ENABLE 1
#else
#define UF_ENABLE 0
#endif
#include "pto_macro_matmul.hpp"
#include "pto_macro_fa_softmax.hpp"
#include "pto_macro_fa_gu.hpp"

using namespace pto;

#ifndef FFTS_BUFFER_FLAG_ENUM
#define FFTS_BUFFER_FLAG_ENUM
enum FftsBufferFlag : uint32_t
{
    BUF0_QK_READY,
    BUF0_SM_CONSUMED,
    BUF1_SM_READY,
    BUF1_SV_CONSUMED,
    UPDATE_READY,
    UPDATE_CONSUMED,
    CV_BLOCK_END = 7,
};
#endif

enum : uint32_t
{
    PV_EVENT_ID0 = 2,
    PV_EVENT_ID1 = 3,
};

constexpr int kHeadSize = 128;
constexpr int kCubeS0 = 128;
constexpr int kCubeS1 = 128;
constexpr int kTileS1 = 256;
constexpr int kQkPreload = 4;
constexpr int kFifoSize = 8;
constexpr int kVecCores = 2;
constexpr int kTileFactor = kTileS1 / kCubeS1;
constexpr int kVecS0 = kCubeS0 / kVecCores / kTileFactor;
constexpr int kVecGuRows = kCubeS0 / kVecCores;
constexpr int VEC_CORES = kVecCores;

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

constexpr std::size_t MAX_TILE_L1_BYTES = 512U * 1024U;
constexpr std::size_t MAX_VEC_UB_BYTES = 192U * 1024U;

AICORE inline bool should_wait_consumption(int tile_id)
{
    return tile_id >= kFifoSize;
}

AICORE inline bool should_notify_consumption(int)
{
    return true;
}

AICORE inline int pending_consumption_events(int tiles_processed)
{
    if (tiles_processed <= 0)
        return 0;
    return tiles_processed < kFifoSize ? tiles_processed : kFifoSize;
}

template <typename TileType>
constexpr AICORE std::size_t tile_storage_bytes()
{
    using ElementType = typename TileType::DType;
    return static_cast<std::size_t>(TileType::Rows * TileType::Cols) * sizeof(ElementType);
}

template <typename TileType, std::size_t NumBuffers>
constexpr AICORE std::size_t tile_buffer_total_bytes()
{
    return tile_storage_bytes<TileType>() * NumBuffers;
}

template <typename TileType, std::size_t NumBuffers>
AICORE inline uint32_t assign_tile_buffers(TileType (&tiles)[NumBuffers], uint32_t base_offset)
{
    if constexpr (NumBuffers == 0) {
        return base_offset;
    }

    constexpr std::size_t total_storage_bytes = tile_buffer_total_bytes<TileType, NumBuffers>();
    static_assert(total_storage_bytes <= MAX_TILE_L1_BYTES, "Tile buffer L1 allocation exceeds 512KB");

    for (std::size_t idx = 0; idx < NumBuffers; ++idx) {
        const uint32_t tile_offset = base_offset + static_cast<uint32_t>(idx * tile_storage_bytes<TileType>());
        TASSIGN(tiles[idx], tile_offset);
    }

    return base_offset + static_cast<uint32_t>(total_storage_bytes);
}

template <typename TileA, std::size_t NumA, typename TileB, std::size_t NumB>
AICORE inline uint32_t assign_tile_buffers_union(TileA (&tilesA)[NumA], TileB (&tilesB)[NumB], uint32_t base_offset)
{
    static_assert(NumA == NumB, "Union assignment expects matching buffer counts");
    if constexpr (NumA == 0) {
        return base_offset;
    }

    constexpr std::size_t stride_bytes = (tile_storage_bytes<TileA>() > tile_storage_bytes<TileB>()) ?
                                             tile_storage_bytes<TileA>() :
                                             tile_storage_bytes<TileB>();
    constexpr std::size_t total_storage_bytes = stride_bytes * NumA;
    static_assert(total_storage_bytes <= MAX_VEC_UB_BYTES, "Union tile UB allocation exceeds 192KB");

    for (std::size_t idx = 0; idx < NumA; ++idx) {
        const uint32_t tile_offset = base_offset + static_cast<uint32_t>(idx * stride_bytes);
        TASSIGN(tilesA[idx], tile_offset);
        TASSIGN(tilesB[idx], tile_offset);
    }

    return base_offset + static_cast<uint32_t>(total_storage_bytes);
}

template <typename TileQType, std::size_t NumQ, typename TileKType, std::size_t NumK, typename TilePType,
          std::size_t NumP, typename TileVType, std::size_t NumV>
AICORE inline void allocate_cube_tile_buffers(TileQType (&qTiles)[NumQ], TileKType (&kTiles)[NumK],
                                              TilePType (&pTiles)[NumP], TileVType (&vTiles)[NumV])
{
    constexpr std::size_t total_bytes =
        tile_buffer_total_bytes<TileQType, NumQ>() + tile_buffer_total_bytes<TileKType, NumK>() +
        tile_buffer_total_bytes<TilePType, NumP>() + tile_buffer_total_bytes<TileVType, NumV>();
    static_assert(total_bytes <= MAX_TILE_L1_BYTES, "Total cube L1 allocation exceeds 512KB");

    uint32_t l1_offset = 0;
    l1_offset = assign_tile_buffers(qTiles, l1_offset);
    l1_offset = assign_tile_buffers(kTiles, l1_offset);
    l1_offset = assign_tile_buffers(pTiles, l1_offset);
    l1_offset = assign_tile_buffers(vTiles, l1_offset);
    (void)l1_offset;
}

template <typename TileDataF_T, typename ReduceTileF_T, typename TileDataH_T, typename TileOutT, std::size_t SrcBuffers,
          std::size_t XexpBuffers, std::size_t pvVecBuffers, std::size_t ExpMaxBuffers>
AICORE inline void allocate_vec_tile_buffers(TileDataF_T (&srcTiles)[SrcBuffers], ReduceTileF_T &m1_local_max,
                                             TileDataF_T &input_reduce_tmp, ReduceTileF_T &l1_local_sum,
                                             ReduceTileF_T &m2_global_max, ReduceTileF_T &l2_global_sum,
                                             ReduceTileF_T (&l1_exp_max)[ExpMaxBuffers],
                                             TileDataH_T (&x_expT)[XexpBuffers], TileOutT (&pvTile)[pvVecBuffers],
                                             TileOutT &runningOTile, TileDataF_T &triu)
{
    constexpr std::size_t float_tile_bytes = tile_storage_bytes<TileDataF_T>();
    constexpr std::size_t reduce_tile_bytes = tile_storage_bytes<ReduceTileF_T>();
    constexpr std::size_t xexp_bytes = tile_buffer_total_bytes<TileDataH_T, XexpBuffers>();
    constexpr std::size_t out_tile_bytes = tile_storage_bytes<TileOutT>();
    constexpr std::size_t union_stride = (tile_storage_bytes<TileDataF_T>() > tile_storage_bytes<TileOutT>()) ?
                                             tile_storage_bytes<TileDataF_T>() :
                                             tile_storage_bytes<TileOutT>();
    static_assert(SrcBuffers == pvVecBuffers, "src/pv ping-pong buffer counts must match for union allocation");
    constexpr std::size_t union_bytes = union_stride * SrcBuffers;
    constexpr std::size_t total_bytes = union_bytes + xexp_bytes + (reduce_tile_bytes * (3U + ExpMaxBuffers)) +
                                        (float_tile_bytes / 8 * 1U) + (float_tile_bytes * 1U) + out_tile_bytes;
    static_assert(total_bytes <= MAX_VEC_UB_BYTES, "Vec tile UB allocation exceeds 192KB");

    uint32_t offset = 0;
    offset = assign_tile_buffers_union(srcTiles, pvTile, offset);

    TASSIGN(m1_local_max, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    TASSIGN(m2_global_max, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    uint32_t tmp_float_offset = offset;
    TASSIGN(input_reduce_tmp, tmp_float_offset);
    offset += static_cast<uint32_t>(float_tile_bytes) / 8;

    TASSIGN(triu, offset);
    offset += static_cast<uint32_t>(float_tile_bytes);

    TASSIGN(l1_local_sum, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    TASSIGN(l2_global_sum, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    offset = assign_tile_buffers(l1_exp_max, offset);

    uint32_t tail_offset = assign_tile_buffers(x_expT, offset);

    TASSIGN(runningOTile, tail_offset);

    tail_offset += static_cast<uint32_t>(out_tile_bytes);
    (void)tail_offset;
}

// Helper to assign an accumulator tile to one of two ping-pong UB addresses (0x0 / 0x10000).
// Keeps a per-type static running index that toggles on every call. Caller may pass
// `initial_id` (0 or 1) to set the starting buffer index on the first call for that tile type.
template <typename AccTileT>
AICORE inline int assign_running_acc_tile(AccTileT &accTile, int initial_id = -1)
{
    static int running_tile_buffer_idx = 0; // per-instantiation running buffer index: 0 -> base0, 1 -> base1
    if (initial_id == 0 || initial_id == 1) {
        running_tile_buffer_idx = initial_id;
    }
    const int id = running_tile_buffer_idx;
    const uint32_t base_addr = (id == 0) ? 0x0u : 0x10000u;
    TASSIGN(accTile, base_addr);
    running_tile_buffer_idx ^= 1; // toggle for next call
    return id;
}

template <int HEAD_SIZE, int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO, bool CAUSAL_MASK,
          typename TileMatQData, typename TileMatKData, typename TileQKData, typename TSyncQK2SM>
AICORE inline void compute_qk(int tile_id, int sub_tile_id, __gm__ half *q, __gm__ half *k, __gm__ float *qk_tile_fifo,
                              TileMatQData &qMatTile, TileMatKData &kMatTile, TileQKData &qkAccTile,
                              uint64_t qkMatTileEventId, int accTileEvtID, TSyncQK2SM &qk2smSync, int blk_idx)
{
    if constexpr (DAV_CUBE) {
        constexpr uint32_t Cube_S0 = CUBE_S0;
        constexpr uint32_t Cube_S1 = CUBE_S1;
        constexpr uint32_t Tile_S1 = TILE_S1;
        constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
        constexpr uint32_t Cube_HEAD = HEAD_SIZE;
        static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
        static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");

        const int s0_index = blk_idx * CUBE_S0;
        const int s1_index = tile_id * static_cast<int>(Tile_S1) + sub_tile_id * static_cast<int>(Cube_S1);
        const int sync_iter = tile_id;
        const bool should_wait_consume = should_wait_consumption(sync_iter);
        if constexpr (CAUSAL_MASK) {
            if (s1_index > s0_index) {
                if (sub_tile_id == 0 && should_wait_consume)
                    qk2smSync.allocate(); // wait for SM consume data
                if (sub_tile_id == static_cast<int>(kTileFactor) - 1)
                    qk2smSync.record(); // notify for QK produce data
                return;
            }
        }
        using GlobalDataQ =
            GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
        using GlobalDataK = GlobalTensor<half, pto::Shape<1, 1, 1, HEAD_SIZE, Cube_S1>,
                                         pto::Stride<1, 1, 1, 1, HEAD_SIZE>, Layout::DN>; // BNSD - (N, K) layout

        GlobalDataQ qGlobal(q);
        GlobalDataK kGlobal(k + s1_index * HEAD_SIZE);

        wait_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);

        if (tile_id == 0 && sub_tile_id == 0) {
            TLOAD(qMatTile, qGlobal);
        }

        TLOAD(kMatTile, kGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

#if UF_ENABLE
        pto_macro_matmul<Cube_S0, Cube_HEAD, Cube_S1>(qMatTile, kMatTile, qkAccTile, AccMode::InitFinalSum);
#else
        wait_flag(PIPE_FIX, PIPE_M, accTileEvtID);
        pto_macro_matmul<Cube_S0, Cube_HEAD, Cube_S1>(qMatTile, kMatTile, qkAccTile, AccMode::Init);
#endif

        set_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);
#if !UF_ENABLE
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

        if (sub_tile_id == 0 && should_wait_consume)
            qk2smSync.allocate(); // wait for SM consume data

        using GlobalDataQK =
            GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
        const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
        const size_t base_elems =
            static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) * static_cast<size_t>(Cube_S0) *
                static_cast<size_t>(Cube_S1) +
            static_cast<size_t>(sub_tile_id) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
        GlobalDataQK qkGlobalTile(qk_tile_fifo + base_elems);

#if UF_ENABLE
        TSTORE<STPhase::Final>(qkGlobalTile, qkAccTile);
#else
        TSTORE(qkGlobalTile, qkAccTile);
        set_flag(PIPE_FIX, PIPE_M, accTileEvtID);
#endif

        if (sub_tile_id == static_cast<int>(kTileFactor) - 1)
            qk2smSync.record(); // notify for QK produce data
    }
}

template <int HEAD_SIZE, int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO, int PV_CV_FIFO,
          typename TileMatPData, typename TileMatVData, typename TilePVData, typename PPipe, typename TSyncPV2GU>
AICORE inline void compute_pv(int tile_id, int sub_tile_id, __gm__ half *v, __gm__ float *pv_tile_fifo,
                              TileMatPData &pMatTile, TileMatVData &vMatTile, TilePVData &pvAccTile,
                              uint64_t svMatTileEventId, int accTileEvtID, PPipe &pPipe, TSyncPV2GU &pv2guSync)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;
    constexpr uint32_t TileElems = Cube_S0 * Tile_S1;
    static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");

    const int s1_index = tile_id * static_cast<int>(Tile_S1) + sub_tile_id * static_cast<int>(Cube_S1);
    const int sync_iter = tile_id;
    const bool should_wait_consume = should_wait_consumption(sync_iter);
    const bool should_notify_consume = should_notify_consumption(sync_iter);
    const bool is_last_subtile = (sub_tile_id + 1 == static_cast<int>(kTileFactor));

    if constexpr (DAV_CUBE) {
        using GlobalVT =
            GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S1, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;

        wait_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

        GlobalVT vLoad((__gm__ half *)(v + s1_index * HEAD_SIZE));
        TLOAD(vMatTile, vLoad);

        bool isWait = (sub_tile_id == 0);
        bool isFree = (sub_tile_id == static_cast<int>(kTileFactor) - 1 && should_notify_consume);
        pPipe.cons.setWaitStatus(isWait);
        pPipe.cons.setFreeStatus(isFree);
        pPipe.cons.setEntryOffset(sub_tile_id * Cube_S0 * Cube_S1 * sizeof(half));
        TPOP(pMatTile, pPipe);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

#if !UF_ENABLE
        if (sub_tile_id == 0) {
            wait_flag(PIPE_FIX, PIPE_M, accTileEvtID);
        }
#endif

#if UF_ENABLE
        const AccMode accMode = (sub_tile_id == 0) ?
                                    (is_last_subtile ? AccMode::InitFinalSum : AccMode::InitPartialSum) :
                                    (is_last_subtile ? AccMode::AccFinalSum : AccMode::AccPartialSum);
        pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD>(pMatTile, vMatTile, pvAccTile, accMode);
#else
        const AccMode accMode = (sub_tile_id == 0) ? AccMode::Init : AccMode::Acc;
        pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD>(pMatTile, vMatTile, pvAccTile, accMode);
#endif

        set_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

        if (sub_tile_id == static_cast<int>(kTileFactor) - 1) {
#if !UF_ENABLE
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
#endif

            if (should_wait_consume)
                pv2guSync.allocate(); // wait for update consume data

            using GlobalDataPV =
                GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
            const uint32_t buf_idx_pv = static_cast<uint32_t>(tile_id % PV_CV_FIFO);
            const size_t base_elems_pv =
                static_cast<size_t>(buf_idx_pv) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);
            GlobalDataPV pvGlobalTile((__gm__ float *)(pv_tile_fifo + base_elems_pv));

#if UF_ENABLE
            TSTORE<STPhase::Final>(pvGlobalTile, pvAccTile);
#else
            TSTORE(pvGlobalTile, pvAccTile);
            set_flag(PIPE_FIX, PIPE_M, accTileEvtID);
#endif

            pv2guSync.record(); // notify update produce data
        }                       // end loop
    }                           // end if DAV_CUBE
}

template <int HEAD_SIZE, int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO, bool CAUSAL_MASK,
          typename TileDataF_T, typename TileDataH_T, typename ReduceTileF_T, typename TSyncQK2SM, typename PPipe>
AICORE inline void compute_p(int tile_id, int row_slice, __gm__ float *qk_tile_fifo, TileDataF_T &qkVecTile,
                             TileDataH_T &x_expT, TileDataF_T &input_reduce_tmp, ReduceTileF_T &m1_local_max,
                             ReduceTileF_T &l1_local_sum, ReduceTileF_T &m2_global_max, ReduceTileF_T &l2_global_sum,
                             ReduceTileF_T &l1_exp_max_ififo, TileDataF_T triu, uint64_t pTileEventId,
                             TSyncQK2SM &qk2smSync, PPipe &pPipe, int blk_idx)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
    const bool initFlag = (tile_id == 0);
    static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");
    static_assert(Cube_S0 % (VEC_CORES * kTileFactor) == 0, "Vec rows must divide evenly across tile slices");
    if constexpr (DAV_VEC) {
        const size_t subblock_base_rows =
            static_cast<size_t>(Cube_S0 / VEC_CORES) * static_cast<size_t>(get_subblockid());
        const size_t row_offset = subblock_base_rows + static_cast<size_t>(row_slice * Vec_S0);
        const int s0_index = blk_idx * Cube_S0 + row_offset;
        const int s1_index = tile_id * static_cast<int>(Tile_S1);
        const int sync_iter = tile_id;
        const bool should_wait_consume = should_wait_consumption(sync_iter);
        const bool should_notify_consume = should_notify_consumption(sync_iter);

        wait_flag(PIPE_V, PIPE_MTE2, pTileEventId);
        if (row_slice == 0)
            qk2smSync.wait(); // wait for QK produce data

        const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
        const size_t base_elems = static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) *
                                  static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
        __gm__ float *qk_ptr = qk_tile_fifo + base_elems + row_offset * static_cast<size_t>(Cube_S1);

        using GlobalDataQK_Sub =
            GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
        using TileDataF_Sub = Tile<TileType::Vec, float, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Cube_S1>;
        for (int sub_col = 0; sub_col < static_cast<int>(kTileFactor); ++sub_col) {
            __gm__ float *qk_ptr_sub =
                qk_ptr + static_cast<size_t>(sub_col) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
            GlobalDataQK_Sub qkGlobalSub(qk_ptr_sub);

            TileDataF_Sub qkVecSub;
            const uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * Cube_S1 * sizeof(float));
            TASSIGN(qkVecSub, (uint64_t)qkVecTile.data() + col_byte_offset);
            TLOAD(qkVecSub, qkGlobalSub);
        }

        if (row_slice == static_cast<int>(kTileFactor) - 1 && should_notify_consume)
            qk2smSync.free(); // notify for SM consume data

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Extract per-slice views into the per-core reduce tiles so each slice writes into its row range
        using ReduceSliceTile = Tile<TileType::Vec, float, Vec_S0, 1, BLayout::ColMajor, Vec_S0, 1>;
        // reduce tiles live per vector core; offset only by row_slice within the core (no subblock stride)
        const size_t reduce_slice_rows = static_cast<size_t>(row_slice * Vec_S0);
        const uint64_t reduce_row_byte_offset = reduce_slice_rows * sizeof(float);

        ReduceSliceTile m1_local_max_slice;
        ReduceSliceTile l1_local_sum_slice;
        ReduceSliceTile m2_global_max_slice;
        ReduceSliceTile l2_global_sum_slice;
        ReduceSliceTile l1_exp_max_slice;

        TASSIGN(m1_local_max_slice, (uint64_t)m1_local_max.data() + reduce_row_byte_offset);
        TASSIGN(l1_local_sum_slice, (uint64_t)l1_local_sum.data() + reduce_row_byte_offset);
        TASSIGN(m2_global_max_slice, (uint64_t)m2_global_max.data() + reduce_row_byte_offset);
        TASSIGN(l2_global_sum_slice, (uint64_t)l2_global_sum.data() + reduce_row_byte_offset);
        TASSIGN(l1_exp_max_slice, (uint64_t)l1_exp_max_ififo.data() + reduce_row_byte_offset);

        // Extract current slice state from full-length reduce tiles
        // Please change to TEXTRACT when available

        wait_flag(PIPE_MTE3, PIPE_V, pTileEventId);
        if (initFlag) {
            pto_macro_fa_softmax<true, HEAD_SIZE, CAUSAL_MASK>(
                x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice, m2_global_max_slice, l2_global_sum_slice,
                l1_exp_max_slice, input_reduce_tmp, qkVecTile, triu, s0_index, s1_index);
        } else {
            pto_macro_fa_softmax<false, HEAD_SIZE, CAUSAL_MASK>(
                x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice, m2_global_max_slice, l2_global_sum_slice,
                l1_exp_max_slice, input_reduce_tmp, qkVecTile, triu, s0_index, s1_index);
        }

        set_flag(PIPE_V, PIPE_MTE2, pTileEventId);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        bool isAllocate = (row_slice == 0 && should_wait_consume);
        const bool isRecord = (row_slice == (static_cast<int>(kTileFactor) - 1));
        pPipe.prod.setAllocateStatus(isAllocate);
        pPipe.prod.setRecordStatus(isRecord);
        pPipe.prod.setEntryOffset(row_offset * Cube_S1 * sizeof(half));
        TPUSH(x_expT, pPipe);

        set_flag(PIPE_MTE3, PIPE_V, pTileEventId);
    }
}

template <int HEAD_SIZE, int CUBE_S0, int TILE_S1, int PV_CV_FIFO, typename TileOutT, typename ReduceTileF_T,
          typename TSyncPV2GU>
AICORE inline void compute_gu(int tile_id, int num_tiles, __gm__ float *pv_tile_fifo, __gm__ float *o_out,
                              TileOutT &runningOTile, TileOutT &pvVecTile, ReduceTileF_T &l1_exp_max_ififo,
                              ReduceTileF_T &l2_global_sum, uint64_t guEventId, TSyncPV2GU &pv2guSync)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES;

    using GlobalDataPV_VEC =
        GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;

    if constexpr (DAV_VEC) {
        const uint32_t buf_idx = static_cast<uint32_t>(tile_id % PV_CV_FIFO);
        const size_t base_elems =
            static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);

        const size_t subblock_base_rows =
            static_cast<size_t>(Cube_S0 / VEC_CORES) * static_cast<size_t>(get_subblockid());
        __gm__ float *pv_out_ptr = pv_tile_fifo + base_elems + subblock_base_rows * HEAD_SIZE;
        GlobalDataPV_VEC pvGlobalVec(pv_out_ptr);

        pv2guSync.wait(); // wait for update consume data

        // softamx output and gu input buffer reuse
        const bool should_notify_consume = should_notify_consumption(tile_id);

        wait_flag(PIPE_V, PIPE_MTE2, guEventId);

        if (tile_id == 0) {
            TLOAD(runningOTile, pvGlobalVec);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        } else {
            TLOAD(pvVecTile, pvGlobalVec);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if (tile_id < num_tiles - 1) {
                pto_macro_fa_gu<ReduceTileF_T, TileOutT>(runningOTile, pvVecTile, l1_exp_max_ififo);
            } else {
                pto_macro_fa_gu_last<ReduceTileF_T, TileOutT>(runningOTile, pvVecTile, l1_exp_max_ififo, l2_global_sum);
            }
        }

        set_flag(PIPE_V, PIPE_MTE2, guEventId);
        if (should_notify_consume)
            pv2guSync.free(); // notify update consume data

        if (tile_id == num_tiles - 1) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            using GlobalOutT =
                GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
            GlobalOutT outGlobal((__gm__ float *)(o_out + subblock_base_rows * HEAD_SIZE));
            TSTORE(outGlobal, runningOTile);
        }
    }
}

template <int HEAD_SIZE, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
          bool CAUSAL_MASK>
__global__ AICORE void runTFA(__gm__ uint64_t *ffts_addr, __gm__ half *q, __gm__ half *k, __gm__ half *v,
                              __gm__ half *p_tile_fifo, __gm__ float *o_out, __gm__ float *qk_tile_fifo,
                              __gm__ float *pv_tile_fifo, int s0, int s1)
{
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask(-1, -1);
    }

    set_ffts_base_addr((uint64_t)ffts_addr);
    if constexpr (DAV_CUBE) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    }

    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1; // sub-tiles per TILE_S1
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
    constexpr uint32_t VecGuRows = Cube_S0 / VEC_CORES;
    static_assert(Cube_S0 % (VEC_CORES * kTileFactor) == 0, "Vec rows must divide evenly across tile slices");

    constexpr uint32_t qkPreloadNum = QK_PRELOAD;

    constexpr uint32_t srcVecTNBuffers = 2;
    constexpr uint32_t xexpVecTNBuffers = 2;
    constexpr uint32_t outOTileNBuffers = 2;
    constexpr uint32_t qMatTNBuffers = 1;
    constexpr uint32_t kMatTNBuffers = 2;
    constexpr uint32_t pMatTNBuffers = 2;
    constexpr uint32_t vMatTNBuffers = 2;
    constexpr uint32_t qkp_tile_fifo_size = CV_FIFO_SIZE;
    constexpr uint32_t pv_tile_fifo_size = CV_FIFO_SIZE;
    static_assert(qkPreloadNum >= 1, "qkPreloadNum must be >= 1");
    static_assert((qkPreloadNum > 1) || (kTileFactor == 1), "qkPreloadNum must be > 1 unless kTileFactor == 1");

    using TileMatQData =
        Tile<TileType::Mat, half, Cube_S0, HEAD_SIZE, BLayout::ColMajor, Cube_S0, HEAD_SIZE, SLayout::RowMajor, 512>;
    using TileMatKData =
        Tile<TileType::Mat, half, HEAD_SIZE, Cube_S1, BLayout::RowMajor, HEAD_SIZE, Cube_S1, SLayout::ColMajor, 512>;
    using TileQKData = TileAcc<float, Cube_S0, Cube_S1, Cube_S0, Cube_S1>;

    TileMatQData qMatTile[qMatTNBuffers];
    TileMatKData kMatTile[kMatTNBuffers];
    TileQKData qkAccTile;

    using TileMatPData =
        Tile<TileType::Mat, half, Cube_S0, Cube_S1, BLayout::ColMajor, Cube_S0, Cube_S1, SLayout::RowMajor, 512>;
    using TileMatVData =
        Tile<TileType::Mat, half, Cube_S1, HEAD_SIZE, BLayout::ColMajor, Cube_S1, HEAD_SIZE, SLayout::RowMajor, 512>;
    using TilePVData = TileAcc<float, Cube_S0, HEAD_SIZE, Cube_S0, HEAD_SIZE>;

    TileMatPData pMatTile[pMatTNBuffers];
    TileMatVData vMatTile[vMatTNBuffers];
    TilePVData pvAccTile;

    allocate_cube_tile_buffers(qMatTile, kMatTile, pMatTile, vMatTile);

    assign_running_acc_tile(qkAccTile, 0);
    assign_running_acc_tile(pvAccTile, 1);

    using TileDataF_T = Tile<TileType::Vec, float, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Tile_S1>;
    using TileDataH_T = Tile<TileType::Vec, half, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Tile_S1>;
    constexpr uint32_t SubblockRows = Cube_S0 / VEC_CORES;
    using ReduceTileF_T = Tile<TileType::Vec, float, SubblockRows, 1, BLayout::ColMajor, SubblockRows, 1>;

    TileDataF_T qkVecTile[srcVecTNBuffers];
    ReduceTileF_T m1_local_max;
    TileDataF_T input_reduce_tmp;
    TileDataF_T triu;
    ReduceTileF_T l1_local_sum;
    ReduceTileF_T m2_global_max;
    ReduceTileF_T l2_global_sum;
    ReduceTileF_T l1_exp_max_ififo[qkp_tile_fifo_size];
    TileDataH_T x_expT[xexpVecTNBuffers];

    using TileOutGuT = Tile<TileType::Vec, float, VecGuRows, HEAD_SIZE, BLayout::RowMajor, VecGuRows, HEAD_SIZE>;
    TileOutGuT pvVecTile[outOTileNBuffers];
    TileOutGuT runningOTile;
    allocate_vec_tile_buffers<TileDataF_T, ReduceTileF_T, TileDataH_T, TileOutGuT, srcVecTNBuffers, xexpVecTNBuffers,
                              outOTileNBuffers>(qkVecTile, m1_local_max, input_reduce_tmp, l1_local_sum, m2_global_max,
                                                l2_global_sum, l1_exp_max_ififo, x_expT, pvVecTile, runningOTile, triu);

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__) // A5 defined macro, don't need to reassign
    const int block_idx = get_block_idx();
#endif
    const int block_offset_rows = block_idx * static_cast<int>(Cube_S0);
    const size_t p_fifo_block_stride =
        static_cast<size_t>(qkp_tile_fifo_size) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Tile_S1);
    const size_t qk_fifo_block_stride = p_fifo_block_stride;
    const size_t pv_fifo_block_stride =
        static_cast<size_t>(pv_tile_fifo_size) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);

    __gm__ half *q_block = q + block_offset_rows * HEAD_SIZE;
    __gm__ half *p_tile_fifo_block = p_tile_fifo + static_cast<size_t>(block_idx) * p_fifo_block_stride;
    __gm__ float *o_out_block = o_out + static_cast<size_t>(block_offset_rows) * static_cast<size_t>(HEAD_SIZE);
    __gm__ float *qk_tile_fifo_block = qk_tile_fifo + static_cast<size_t>(block_idx) * qk_fifo_block_stride;
    __gm__ float *pv_tile_fifo_block = pv_tile_fifo + static_cast<size_t>(block_idx) * pv_fifo_block_stride;

    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> qk2smSync = {BUF0_QK_READY};
    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> pv2guSync = {UPDATE_READY};

    constexpr uint8_t FiFoDepth = qkp_tile_fifo_size;
    constexpr uint8_t FiFoSyncT = 1;
    using PPipe = TMPipe<BUF1_SM_READY, FIFOType::GM_FIFO, FiFoDepth, FiFoSyncT, TileDataH_T, TileMatPData, false, 0>;
    PPipe pPipe(p_tile_fifo_block, (uint32_t)(uint64_t)pMatTile[0].data());

    int num_tiles_s1 = s1 / Tile_S1;
    if constexpr (DAV_CUBE) {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
    }
    if constexpr (DAV_VEC) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    }

    int p_gu_src_pingpong_id = 0; // shared ping-pong for softmax vec tiles, pv output tiles, and GU input tiles
    int k_src_pingpong_id = 0;    // separate ping-pong for K tiles
    int pv_src_pingpong_id = 0;   // separate ping-pong for P V tiles

    int qkAccTileEvtID = 0;
    int pvAccTileEvtID = 0;

    // QK and P pre-computation (tile_id based)
    for (int preload_tile = 0; preload_tile < static_cast<int>(qkPreloadNum) && preload_tile < num_tiles_s1;
         ++preload_tile) {
        if constexpr (DAV_CUBE) {
            for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor); ++sub_tile) {
                qkAccTileEvtID = assign_running_acc_tile(qkAccTile);
                compute_qk<HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, CAUSAL_MASK>(
                    preload_tile, sub_tile, q_block, k, qk_tile_fifo_block, qMatTile[0],
                    kMatTile[k_src_pingpong_id % kMatTNBuffers], qkAccTile, k_src_pingpong_id % kMatTNBuffers,
                    qkAccTileEvtID, qk2smSync, block_idx);
                k_src_pingpong_id++;
            }
        }
        if constexpr (DAV_VEC) {
            for (int row_slice = 0; row_slice < static_cast<int>(kTileFactor); ++row_slice) {
                // Init only on the very first S1 tile; row_slice partitions rows within that tile
                pPipe.prod.setTileId(preload_tile, row_slice);
                compute_p<HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, CAUSAL_MASK>(
                    preload_tile, row_slice, qk_tile_fifo_block,
                    qkVecTile[p_gu_src_pingpong_id % srcVecTNBuffers], x_expT[p_gu_src_pingpong_id % xexpVecTNBuffers],
                    input_reduce_tmp, m1_local_max, l1_local_sum, m2_global_max, l2_global_sum,
                    l1_exp_max_ififo[preload_tile % qkp_tile_fifo_size], triu, p_gu_src_pingpong_id % xexpVecTNBuffers,
                    qk2smSync, pPipe, block_idx);
                p_gu_src_pingpong_id++;
            }
        }
    }

    for (int tile_id = 0; tile_id < num_tiles_s1; ++tile_id) {
        int next_qk_tile = (tile_id + static_cast<int>(qkPreloadNum) >= num_tiles_s1) ?
                               -1 :
                               (tile_id + static_cast<int>(qkPreloadNum));

        if (next_qk_tile != -1)
            qkAccTileEvtID = assign_running_acc_tile(qkAccTile);
        pvAccTileEvtID = assign_running_acc_tile(pvAccTile);

        for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor); ++sub_tile) {
            if constexpr (DAV_CUBE) {
                if (next_qk_tile != -1) {
                    compute_qk<HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, CAUSAL_MASK>(
                        next_qk_tile, sub_tile, q_block, k, qk_tile_fifo_block, qMatTile[0],
                        kMatTile[k_src_pingpong_id % kMatTNBuffers], qkAccTile, k_src_pingpong_id % kMatTNBuffers,
                        qkAccTileEvtID, qk2smSync, block_idx);
                    k_src_pingpong_id++;
                }
            }

            if constexpr (DAV_VEC) {
                if (next_qk_tile != -1) {
                    pPipe.prod.setTileId(next_qk_tile, sub_tile);
                    compute_p<HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, CAUSAL_MASK>(
                        next_qk_tile, sub_tile, qk_tile_fifo_block, qkVecTile[p_gu_src_pingpong_id % srcVecTNBuffers],
                        x_expT[p_gu_src_pingpong_id % xexpVecTNBuffers], input_reduce_tmp, m1_local_max, l1_local_sum,
                        m2_global_max, l2_global_sum, l1_exp_max_ififo[next_qk_tile % qkp_tile_fifo_size], triu,
                        p_gu_src_pingpong_id % xexpVecTNBuffers, qk2smSync, pPipe, block_idx);
                    p_gu_src_pingpong_id++;
                }
            }

            if constexpr (DAV_CUBE) {
                pPipe.cons.setTileId(tile_id, sub_tile);
                compute_pv<HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, pv_tile_fifo_size>(
                    tile_id, sub_tile, v, pv_tile_fifo_block, pMatTile[pv_src_pingpong_id % pMatTNBuffers],
                    vMatTile[pv_src_pingpong_id % vMatTNBuffers], pvAccTile,
                    pv_src_pingpong_id % vMatTNBuffers + PV_EVENT_ID0, pvAccTileEvtID, pPipe, pv2guSync);
                pv_src_pingpong_id++;
            }
        }

        if constexpr (DAV_VEC) {
            compute_gu<HEAD_SIZE, CUBE_S0, Tile_S1, pv_tile_fifo_size>(
                tile_id, num_tiles_s1, pv_tile_fifo_block, o_out_block, runningOTile,
                pvVecTile[p_gu_src_pingpong_id % outOTileNBuffers], l1_exp_max_ififo[tile_id % qkp_tile_fifo_size],
                l2_global_sum, p_gu_src_pingpong_id % outOTileNBuffers, pv2guSync);
            p_gu_src_pingpong_id++;
        }
    }

    const int pending_qk_sm_consumed = pending_consumption_events(num_tiles_s1);
    const int pending_sv_consumed = pending_qk_sm_consumed; // same schedule and FIFO settings
    const int pending_update_consumed = pending_consumption_events(num_tiles_s1);

    if constexpr (DAV_CUBE) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        for (int i = 0; i < pending_qk_sm_consumed; ++i)
            qk2smSync.allocate();
        for (int i = 0; i < pending_update_consumed; ++i)
            pv2guSync.allocate();
#ifdef __DAV_C220_CUBE__
        wait_flag_dev(CV_BLOCK_END);
#endif
    }

    if constexpr (DAV_VEC) {
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        for (int i = 0; i < pending_sv_consumed; ++i)
            pPipe.prod.allocate();
#ifdef __DAV_C220_VEC__
        ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(CV_CORE_SYNC, CV_BLOCK_END));
#endif
    }

    pipe_barrier(PIPE_ALL);
}

extern "C" void call_kernel(void *stream,

                            // parameters
                            int headSize, // must be 128
                            int s0, int s1, bool is_causal,

                            // inputs
                            uint8_t *q, uint8_t *k, uint8_t *v,

                            // final output
                            uint8_t *o_out,

                            // -------- workspace / intermediates --------
                            float *outDevice,      // qk_tile_fifo
                            uint16_t *xexpDevice,  // p_tile_fifo (half)
                            float *pOutFp32Device,
                            float *out2Device,     // pv_tile_fifo
                            float *gSumDevice,
                            float *expMaxDevice,
                            float *oPartsDevice
)
{
    (void)pOutFp32Device;
    (void)gSumDevice;
    (void)expMaxDevice;
    (void)oPartsDevice;

    if (headSize != kHeadSize || (s0 % kCubeS0) != 0 || (s1 % kTileS1) != 0)
        return;

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);
    (void)fftsLen;

    const int blocks = s0 / kCubeS0;
    if (is_causal) {
        runTFA<kHeadSize, kCubeS0, kCubeS1, kTileS1, kQkPreload, kFifoSize, true>
            <<<blocks, nullptr, stream>>>((__gm__ uint64_t *)ffts, (__gm__ half *)q, (__gm__ half *)k,
                                          (__gm__ half *)v, (__gm__ half *)xexpDevice, (__gm__ float *)o_out,
                                          (__gm__ float *)outDevice, (__gm__ float *)out2Device, s0, s1);
    } else {
        runTFA<kHeadSize, kCubeS0, kCubeS1, kTileS1, kQkPreload, kFifoSize, false>
            <<<blocks, nullptr, stream>>>((__gm__ uint64_t *)ffts, (__gm__ half *)q, (__gm__ half *)k,
                                          (__gm__ half *)v, (__gm__ half *)xexpDevice, (__gm__ float *)o_out,
                                          (__gm__ float *)outDevice, (__gm__ float *)out2Device, s0, s1);
    }
}
