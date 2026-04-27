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
#include <pto/npu/a2a3/custom/TSyncCVID.hpp>
#include <pto/npu/a2a3/custom/TSync_Custom.hpp>
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
constexpr int kSrcVecBuffers = 2;
constexpr int kXexpVecBuffers = 2;
constexpr int kOutVecBuffers = 2;
constexpr int kQMatBuffers = 1;
constexpr int kKMatBuffers = 2;
constexpr int kPMatBuffers = 2;
constexpr int kVMatBuffers = 2;

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

AICORE inline int pending_consumption_events(int tiles_processed)
{
    if (tiles_processed <= 0)
        return 0;
    return tiles_processed < kFifoSize ? tiles_processed : kFifoSize;
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

using TileMatQData =
    Tile<TileType::Mat, half, kCubeS0, kHeadSize, BLayout::ColMajor, kCubeS0, kHeadSize, SLayout::RowMajor, 512>;
using TileMatKData =
    Tile<TileType::Mat, half, kHeadSize, kCubeS1, BLayout::RowMajor, kHeadSize, kCubeS1, SLayout::ColMajor, 512>;
using TileQKData = TileAcc<float, kCubeS0, kCubeS1, kCubeS0, kCubeS1>;
using TileMatPData =
    Tile<TileType::Mat, half, kCubeS0, kCubeS1, BLayout::ColMajor, kCubeS0, kCubeS1, SLayout::RowMajor, 512>;
using TileMatVData =
    Tile<TileType::Mat, half, kCubeS1, kHeadSize, BLayout::ColMajor, kCubeS1, kHeadSize, SLayout::RowMajor, 512>;
using TilePVData = TileAcc<float, kCubeS0, kHeadSize, kCubeS0, kHeadSize>;
using TileDataF_T = Tile<TileType::Vec, float, kVecS0, kTileS1, BLayout::RowMajor, kVecS0, kTileS1>;
using TileDataH_T = Tile<TileType::Vec, half, kVecS0, kTileS1, BLayout::RowMajor, kVecS0, kTileS1>;
using ReduceTileF_T = Tile<TileType::Vec, float, kVecGuRows, 1, BLayout::ColMajor, kVecGuRows, 1>;
using TileOutGuT = Tile<TileType::Vec, float, kVecGuRows, kHeadSize, BLayout::RowMajor, kVecGuRows, kHeadSize>;

template <bool CAUSAL_MASK, typename TSyncQK2SM>
AICORE inline void compute_qk(int tile_id, int sub_tile_id, __gm__ half *q, __gm__ half *k, __gm__ float *qk_tile_fifo,
                              TileMatQData &qMatTile, TileMatKData &kMatTile, TileQKData &qkAccTile,
                              uint64_t qkMatTileEventId, TSyncQK2SM &qk2smSync, int blk_idx)
{
    if constexpr (DAV_CUBE) {
        const int s0_index = blk_idx * kCubeS0;
        const int s1_index = tile_id * kTileS1 + sub_tile_id * kCubeS1;
        const bool should_wait_consume = should_wait_consumption(tile_id);
        if constexpr (CAUSAL_MASK) {
            if (s1_index > s0_index) {
                if (sub_tile_id == 0 && should_wait_consume)
                    qk2smSync.allocate();
                if (sub_tile_id == kTileFactor - 1)
                    qk2smSync.record();
                return;
            }
        }
        using GlobalDataQ =
            GlobalTensor<half, pto::Shape<1, 1, 1, kCubeS0, kHeadSize>, pto::Stride<1, 1, 1, kHeadSize, 1>>;
        using GlobalDataK = GlobalTensor<half, pto::Shape<1, 1, 1, kHeadSize, kCubeS1>,
                                         pto::Stride<1, 1, 1, 1, kHeadSize>, Layout::DN>;

        GlobalDataQ qGlobal(q);
        GlobalDataK kGlobal(k + s1_index * kHeadSize);

        wait_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);

        if (tile_id == 0 && sub_tile_id == 0) {
            TLOAD(qMatTile, qGlobal);
        }

        TLOAD(kMatTile, kGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        pto_macro_matmul<kCubeS0, kHeadSize, kCubeS1>(qMatTile, kMatTile, qkAccTile, AccMode::InitFinalSum);

        set_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);

        if (sub_tile_id == 0 && should_wait_consume)
            qk2smSync.allocate();

        using GlobalDataQK =
            GlobalTensor<float, pto::Shape<1, 1, 1, kCubeS0, kCubeS1>, pto::Stride<1, 1, 1, kCubeS1, 1>>;
        const size_t base_elems = static_cast<size_t>(tile_id % kFifoSize) * kCubeS0 * kTileS1 +
                                  static_cast<size_t>(sub_tile_id) * kCubeS0 * kCubeS1;
        GlobalDataQK qkGlobalTile(qk_tile_fifo + base_elems);

        TSTORE<STPhase::Final>(qkGlobalTile, qkAccTile);

        if (sub_tile_id == kTileFactor - 1)
            qk2smSync.record();
    }
}

template <typename PPipe, typename TSyncPV2GU>
AICORE inline void compute_pv(int tile_id, int sub_tile_id, __gm__ half *v, __gm__ float *pv_tile_fifo,
                              TileMatPData &pMatTile, TileMatVData &vMatTile, TilePVData &pvAccTile,
                              uint64_t svMatTileEventId, PPipe &pPipe, TSyncPV2GU &pv2guSync)
{
    const int s1_index = tile_id * kTileS1 + sub_tile_id * kCubeS1;
    const bool should_wait_consume = should_wait_consumption(tile_id);
    const bool is_last_subtile = sub_tile_id + 1 == kTileFactor;

    if constexpr (DAV_CUBE) {
        using GlobalVT =
            GlobalTensor<half, pto::Shape<1, 1, 1, kCubeS1, kHeadSize>, pto::Stride<1, 1, 1, kHeadSize, 1>>;

        wait_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

        GlobalVT vLoad((__gm__ half *)(v + s1_index * kHeadSize));
        TLOAD(vMatTile, vLoad);

        pPipe.cons.setWaitStatus(sub_tile_id == 0);
        pPipe.cons.setFreeStatus(is_last_subtile);
        pPipe.cons.setEntryOffset(sub_tile_id * kCubeS0 * kCubeS1 * sizeof(half));
        TPOP(pMatTile, pPipe);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        const AccMode accMode = (sub_tile_id == 0) ?
                                    (is_last_subtile ? AccMode::InitFinalSum : AccMode::InitPartialSum) :
                                    (is_last_subtile ? AccMode::AccFinalSum : AccMode::AccPartialSum);
        pto_macro_matmul<kCubeS0, kCubeS1, kHeadSize>(pMatTile, vMatTile, pvAccTile, accMode);

        set_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

        if (is_last_subtile) {
            if (should_wait_consume)
                pv2guSync.allocate();

            using GlobalDataPV =
                GlobalTensor<float, pto::Shape<1, 1, 1, kCubeS0, kHeadSize>, pto::Stride<1, 1, 1, kHeadSize, 1>>;
            const size_t base_elems_pv = static_cast<size_t>(tile_id % kFifoSize) * kCubeS0 * kHeadSize;
            GlobalDataPV pvGlobalTile((__gm__ float *)(pv_tile_fifo + base_elems_pv));

            TSTORE<STPhase::Final>(pvGlobalTile, pvAccTile);

            pv2guSync.record();
        }
    }
}

template <bool CAUSAL_MASK, typename TSyncQK2SM, typename PPipe>
AICORE inline void compute_p(int tile_id, int row_slice, __gm__ float *qk_tile_fifo, TileDataF_T &qkVecTile,
                             TileDataH_T &x_expT, TileDataF_T &input_reduce_tmp, ReduceTileF_T &m1_local_max,
                             ReduceTileF_T &l1_local_sum, ReduceTileF_T &m2_global_max, ReduceTileF_T &l2_global_sum,
                             ReduceTileF_T &l1_exp_max_ififo, TileDataF_T triu, uint64_t pTileEventId,
                             TSyncQK2SM &qk2smSync, PPipe &pPipe, int blk_idx)
{
    const bool initFlag = (tile_id == 0);
    if constexpr (DAV_VEC) {
        const size_t subblock_base_rows =
            static_cast<size_t>(kCubeS0 / kVecCores) * static_cast<size_t>(get_subblockid());
        const size_t row_offset = subblock_base_rows + static_cast<size_t>(row_slice * kVecS0);
        const int s0_index = blk_idx * kCubeS0 + row_offset;
        const int s1_index = tile_id * kTileS1;
        const bool should_wait_consume = should_wait_consumption(tile_id);

        wait_flag(PIPE_V, PIPE_MTE2, pTileEventId);
        if (row_slice == 0)
            qk2smSync.wait();

        const size_t base_elems = static_cast<size_t>(tile_id % kFifoSize) * kCubeS0 * kTileS1;
        __gm__ float *qk_ptr = qk_tile_fifo + base_elems + row_offset * kCubeS1;

        using GlobalDataQK_Sub =
            GlobalTensor<float, pto::Shape<1, 1, 1, kVecS0, kCubeS1>, pto::Stride<1, 1, 1, kCubeS1, 1>>;
        using TileDataF_Sub = Tile<TileType::Vec, float, kVecS0, kTileS1, BLayout::RowMajor, kVecS0, kCubeS1>;
        for (int sub_col = 0; sub_col < kTileFactor; ++sub_col) {
            __gm__ float *qk_ptr_sub = qk_ptr + static_cast<size_t>(sub_col) * kCubeS0 * kCubeS1;
            GlobalDataQK_Sub qkGlobalSub(qk_ptr_sub);

            TileDataF_Sub qkVecSub;
            const uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * kCubeS1 * sizeof(float));
            TASSIGN(qkVecSub, (uint64_t)qkVecTile.data() + col_byte_offset);
            TLOAD(qkVecSub, qkGlobalSub);
        }

        if (row_slice == kTileFactor - 1)
            qk2smSync.free();

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        using ReduceSliceTile = Tile<TileType::Vec, float, kVecS0, 1, BLayout::ColMajor, kVecS0, 1>;
        const size_t reduce_slice_rows = static_cast<size_t>(row_slice * kVecS0);
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
            pto_macro_fa_softmax<true, kHeadSize, CAUSAL_MASK>(
                x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice, m2_global_max_slice, l2_global_sum_slice,
                l1_exp_max_slice, input_reduce_tmp, qkVecTile, triu, s0_index, s1_index);
        } else {
            pto_macro_fa_softmax<false, kHeadSize, CAUSAL_MASK>(
                x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice, m2_global_max_slice, l2_global_sum_slice,
                l1_exp_max_slice, input_reduce_tmp, qkVecTile, triu, s0_index, s1_index);
        }

        set_flag(PIPE_V, PIPE_MTE2, pTileEventId);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        pPipe.prod.setAllocateStatus(row_slice == 0 && should_wait_consume);
        pPipe.prod.setRecordStatus(row_slice == kTileFactor - 1);
        pPipe.prod.setEntryOffset(row_offset * kCubeS1 * sizeof(half));
        TPUSH(x_expT, pPipe);

        set_flag(PIPE_MTE3, PIPE_V, pTileEventId);
    }
}

template <typename TSyncPV2GU>
AICORE inline void compute_gu(int tile_id, int num_tiles, __gm__ float *pv_tile_fifo, __gm__ float *o_out,
                              TileOutGuT &runningOTile, TileOutGuT &pvVecTile, ReduceTileF_T &l1_exp_max_ififo,
                              ReduceTileF_T &l2_global_sum, uint64_t guEventId, TSyncPV2GU &pv2guSync)
{
    using GlobalDataPV_VEC =
        GlobalTensor<float, pto::Shape<1, 1, 1, kVecGuRows, kHeadSize>, pto::Stride<1, 1, 1, kHeadSize, 1>>;

    if constexpr (DAV_VEC) {
        const size_t base_elems = static_cast<size_t>(tile_id % kFifoSize) * kCubeS0 * kHeadSize;
        const size_t subblock_base_rows = static_cast<size_t>(kVecGuRows) * static_cast<size_t>(get_subblockid());
        __gm__ float *pv_out_ptr = pv_tile_fifo + base_elems + subblock_base_rows * kHeadSize;
        GlobalDataPV_VEC pvGlobalVec(pv_out_ptr);

        pv2guSync.wait();

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
                pto_macro_fa_gu<ReduceTileF_T, TileOutGuT>(runningOTile, pvVecTile, l1_exp_max_ififo);
            } else {
                pto_macro_fa_gu_last<ReduceTileF_T, TileOutGuT>(runningOTile, pvVecTile, l1_exp_max_ififo,
                                                                 l2_global_sum);
            }
        }

        set_flag(PIPE_V, PIPE_MTE2, guEventId);
        pv2guSync.free();

        if (tile_id == num_tiles - 1) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            using GlobalOutT =
                GlobalTensor<float, pto::Shape<1, 1, 1, kVecGuRows, kHeadSize>, pto::Stride<1, 1, 1, kHeadSize, 1>>;
            GlobalOutT outGlobal((__gm__ float *)(o_out + subblock_base_rows * kHeadSize));
            TSTORE(outGlobal, runningOTile);
        }
    }
}

template <bool CAUSAL_MASK>
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

    static_assert(kQkPreload >= 1, "QK_PRELOAD must be >= 1");
    static_assert((kQkPreload > 1) || (kTileFactor == 1), "QK_PRELOAD must be > 1 unless kTileFactor == 1");

    TileMatQData qMatTile[kQMatBuffers];
    TileMatKData kMatTile[kKMatBuffers];
    TileQKData qkAccTile;
    TileMatPData pMatTile[kPMatBuffers];
    TileMatVData vMatTile[kVMatBuffers];
    TilePVData pvAccTile;

    constexpr uint32_t mat_tile_bytes = kCubeS0 * kHeadSize * sizeof(half);
    static_assert(mat_tile_bytes * (kQMatBuffers + kKMatBuffers + kPMatBuffers + kVMatBuffers) <=
                      MAX_TILE_L1_BYTES,
                  "Total cube L1 allocation exceeds 512KB");

    uint32_t l1_offset = 0;
    for (int i = 0; i < kQMatBuffers; ++i, l1_offset += mat_tile_bytes)
        TASSIGN(qMatTile[i], l1_offset);
    for (int i = 0; i < kKMatBuffers; ++i, l1_offset += mat_tile_bytes)
        TASSIGN(kMatTile[i], l1_offset);
    for (int i = 0; i < kPMatBuffers; ++i, l1_offset += mat_tile_bytes)
        TASSIGN(pMatTile[i], l1_offset);
    for (int i = 0; i < kVMatBuffers; ++i, l1_offset += mat_tile_bytes)
        TASSIGN(vMatTile[i], l1_offset);

    assign_running_acc_tile(qkAccTile, 0);
    assign_running_acc_tile(pvAccTile, 1);

    TileDataF_T qkVecTile[kSrcVecBuffers];
    ReduceTileF_T m1_local_max;
    TileDataF_T input_reduce_tmp;
    TileDataF_T triu;
    ReduceTileF_T l1_local_sum;
    ReduceTileF_T m2_global_max;
    ReduceTileF_T l2_global_sum;
    ReduceTileF_T l1_exp_max_ififo[kFifoSize];
    TileDataH_T x_expT[kXexpVecBuffers];
    TileOutGuT pvVecTile[kOutVecBuffers];
    TileOutGuT runningOTile;

    constexpr uint32_t vec_tile_bytes = kVecS0 * kTileS1 * sizeof(float);
    constexpr uint32_t xexp_tile_bytes = kVecS0 * kTileS1 * sizeof(half);
    constexpr uint32_t reduce_tile_bytes = kVecGuRows * sizeof(float);
    constexpr uint32_t out_tile_bytes = kVecGuRows * kHeadSize * sizeof(float);
    constexpr uint32_t vec_total_bytes =
        out_tile_bytes * kOutVecBuffers + xexp_tile_bytes * kXexpVecBuffers +
        reduce_tile_bytes * (4U + kFifoSize) + vec_tile_bytes / 8U + vec_tile_bytes + out_tile_bytes;
    static_assert(vec_tile_bytes == out_tile_bytes, "Vec tile union assumes matching tile sizes");
    static_assert(vec_total_bytes <= MAX_VEC_UB_BYTES, "Vec tile UB allocation exceeds 192KB");

    uint32_t ub_offset = 0;
    for (int i = 0; i < kSrcVecBuffers; ++i, ub_offset += out_tile_bytes) {
        TASSIGN(qkVecTile[i], ub_offset);
        TASSIGN(pvVecTile[i], ub_offset);
    }
    TASSIGN(m1_local_max, ub_offset);
    ub_offset += reduce_tile_bytes;
    TASSIGN(m2_global_max, ub_offset);
    ub_offset += reduce_tile_bytes;
    TASSIGN(input_reduce_tmp, ub_offset);
    ub_offset += vec_tile_bytes / 8U;
    TASSIGN(triu, ub_offset);
    ub_offset += vec_tile_bytes;
    TASSIGN(l1_local_sum, ub_offset);
    ub_offset += reduce_tile_bytes;
    TASSIGN(l2_global_sum, ub_offset);
    ub_offset += reduce_tile_bytes;
    for (int i = 0; i < kFifoSize; ++i, ub_offset += reduce_tile_bytes)
        TASSIGN(l1_exp_max_ififo[i], ub_offset);
    for (int i = 0; i < kXexpVecBuffers; ++i, ub_offset += xexp_tile_bytes)
        TASSIGN(x_expT[i], ub_offset);
    TASSIGN(runningOTile, ub_offset);

#if defined(__DAV_C220_CUBE__) || defined(__DAV_C220_VEC__) // A5 defined macro, don't need to reassign
    const int block_idx = get_block_idx();
#endif
    const int block_offset_rows = block_idx * kCubeS0;
    const size_t p_fifo_block_stride =
        static_cast<size_t>(kFifoSize) * static_cast<size_t>(kCubeS0) * static_cast<size_t>(kTileS1);
    const size_t qk_fifo_block_stride = p_fifo_block_stride;
    const size_t pv_fifo_block_stride = static_cast<size_t>(kFifoSize) * kCubeS0 * kHeadSize;

    __gm__ half *q_block = q + block_offset_rows * kHeadSize;
    __gm__ half *p_tile_fifo_block = p_tile_fifo + static_cast<size_t>(block_idx) * p_fifo_block_stride;
    __gm__ float *o_out_block = o_out + static_cast<size_t>(block_offset_rows) * kHeadSize;
    __gm__ float *qk_tile_fifo_block = qk_tile_fifo + static_cast<size_t>(block_idx) * qk_fifo_block_stride;
    __gm__ float *pv_tile_fifo_block = pv_tile_fifo + static_cast<size_t>(block_idx) * pv_fifo_block_stride;

    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> qk2smSync = {BUF0_QK_READY};
    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> pv2guSync = {UPDATE_READY};

    constexpr uint8_t FiFoDepth = kFifoSize;
    constexpr uint8_t FiFoSyncT = 1;
    using PPipe = TMPipe<BUF1_SM_READY, FIFOType::GM_FIFO, FiFoDepth, FiFoSyncT, TileDataH_T, TileMatPData, false, 0>;
    PPipe pPipe(p_tile_fifo_block, (uint32_t)(uint64_t)pMatTile[0].data());

    int num_tiles_s1 = s1 / kTileS1;
    if constexpr (DAV_CUBE) {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
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

    // QK and P pre-computation (tile_id based)
    for (int preload_tile = 0; preload_tile < kQkPreload && preload_tile < num_tiles_s1;
         ++preload_tile) {
        if constexpr (DAV_CUBE) {
            for (int sub_tile = 0; sub_tile < kTileFactor; ++sub_tile) {
                assign_running_acc_tile(qkAccTile);
                compute_qk<CAUSAL_MASK>(
                    preload_tile, sub_tile, q_block, k, qk_tile_fifo_block, qMatTile[0],
                    kMatTile[k_src_pingpong_id % kKMatBuffers], qkAccTile, k_src_pingpong_id % kKMatBuffers,
                    qk2smSync, block_idx);
                k_src_pingpong_id++;
            }
        }
        if constexpr (DAV_VEC) {
            for (int row_slice = 0; row_slice < kTileFactor; ++row_slice) {
                // Init only on the very first S1 tile; row_slice partitions rows within that tile
                pPipe.prod.setTileId(preload_tile, row_slice);
                compute_p<CAUSAL_MASK>(
                    preload_tile, row_slice, qk_tile_fifo_block,
                    qkVecTile[p_gu_src_pingpong_id % kSrcVecBuffers], x_expT[p_gu_src_pingpong_id % kXexpVecBuffers],
                    input_reduce_tmp, m1_local_max, l1_local_sum, m2_global_max, l2_global_sum,
                    l1_exp_max_ififo[preload_tile % kFifoSize], triu, p_gu_src_pingpong_id % kXexpVecBuffers,
                    qk2smSync, pPipe, block_idx);
                p_gu_src_pingpong_id++;
            }
        }
    }

    for (int tile_id = 0; tile_id < num_tiles_s1; ++tile_id) {
        int next_qk_tile = (tile_id + kQkPreload >= num_tiles_s1) ?
                               -1 :
                               (tile_id + kQkPreload);

        if (next_qk_tile != -1)
            assign_running_acc_tile(qkAccTile);
        assign_running_acc_tile(pvAccTile);

        for (int sub_tile = 0; sub_tile < kTileFactor; ++sub_tile) {
            if constexpr (DAV_CUBE) {
                if (next_qk_tile != -1) {
                    compute_qk<CAUSAL_MASK>(
                        next_qk_tile, sub_tile, q_block, k, qk_tile_fifo_block, qMatTile[0],
                        kMatTile[k_src_pingpong_id % kKMatBuffers], qkAccTile, k_src_pingpong_id % kKMatBuffers,
                        qk2smSync, block_idx);
                    k_src_pingpong_id++;
                }
            }

            if constexpr (DAV_VEC) {
                if (next_qk_tile != -1) {
                    pPipe.prod.setTileId(next_qk_tile, sub_tile);
                    compute_p<CAUSAL_MASK>(
                        next_qk_tile, sub_tile, qk_tile_fifo_block, qkVecTile[p_gu_src_pingpong_id % kSrcVecBuffers],
                        x_expT[p_gu_src_pingpong_id % kXexpVecBuffers], input_reduce_tmp, m1_local_max, l1_local_sum,
                        m2_global_max, l2_global_sum, l1_exp_max_ififo[next_qk_tile % kFifoSize], triu,
                        p_gu_src_pingpong_id % kXexpVecBuffers, qk2smSync, pPipe, block_idx);
                    p_gu_src_pingpong_id++;
                }
            }

            if constexpr (DAV_CUBE) {
                pPipe.cons.setTileId(tile_id, sub_tile);
                compute_pv(
                    tile_id, sub_tile, v, pv_tile_fifo_block, pMatTile[pv_src_pingpong_id % kPMatBuffers],
                    vMatTile[pv_src_pingpong_id % kVMatBuffers], pvAccTile,
                    pv_src_pingpong_id % kVMatBuffers + PV_EVENT_ID0, pPipe, pv2guSync);
                pv_src_pingpong_id++;
            }
        }

        if constexpr (DAV_VEC) {
            compute_gu(
                tile_id, num_tiles_s1, pv_tile_fifo_block, o_out_block, runningOTile,
                pvVecTile[p_gu_src_pingpong_id % kOutVecBuffers], l1_exp_max_ififo[tile_id % kFifoSize],
                l2_global_sum, p_gu_src_pingpong_id % kOutVecBuffers, pv2guSync);
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
        runTFA<true>
            <<<blocks, nullptr, stream>>>((__gm__ uint64_t *)ffts, (__gm__ half *)q, (__gm__ half *)k,
                                          (__gm__ half *)v, (__gm__ half *)xexpDevice, (__gm__ float *)o_out,
                                          (__gm__ float *)outDevice, (__gm__ float *)out2Device, s0, s1);
    } else {
        runTFA<false>
            <<<blocks, nullptr, stream>>>((__gm__ uint64_t *)ffts, (__gm__ half *)q, (__gm__ half *)k,
                                          (__gm__ half *)v, (__gm__ half *)xexpDevice, (__gm__ float *)o_out,
                                          (__gm__ float *)outDevice, (__gm__ float *)out2Device, s0, s1);
    }
}
