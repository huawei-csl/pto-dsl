#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>

using namespace pto;

#define VEC_CORES 2

using ExampleInT = half;
using ExampleOutT = float;
constexpr uint32_t EXAMPLE_TOTAL_M = 16;
constexpr uint32_t EXAMPLE_CASE_TILE_M = 16;
constexpr uint32_t EXAMPLE_TILE_K = 32;
constexpr uint32_t EXAMPLE_TILE_N = 32;

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

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

#ifdef TPUSHPOP_SANITY_ONLY
__global__ AICORE void runSanityMatmul(__gm__ ExampleOutT *out, __gm__ ExampleInT *srcA, __gm__ ExampleInT *srcB)
{
    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(ExampleInT);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(EXAMPLE_TOTAL_M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(EXAMPLE_TILE_K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(EXAMPLE_TILE_N, blockAlign);

    using GlobalA =
        GlobalTensor<ExampleInT, pto::Shape<1, 1, 1, EXAMPLE_TOTAL_M, EXAMPLE_TILE_K>,
                     pto::Stride<EXAMPLE_TOTAL_M * EXAMPLE_TILE_K, EXAMPLE_TOTAL_M * EXAMPLE_TILE_K,
                                 EXAMPLE_TOTAL_M * EXAMPLE_TILE_K, EXAMPLE_TILE_K, 1>>;
    using GlobalB =
        GlobalTensor<ExampleInT, pto::Shape<1, 1, 1, EXAMPLE_TILE_K, EXAMPLE_TILE_N>,
                     pto::Stride<EXAMPLE_TILE_K * EXAMPLE_TILE_N, EXAMPLE_TILE_K * EXAMPLE_TILE_N,
                                 EXAMPLE_TILE_K * EXAMPLE_TILE_N, EXAMPLE_TILE_N, 1>>;
    using GlobalOut =
        GlobalTensor<ExampleOutT, pto::Shape<1, 1, 1, EXAMPLE_TOTAL_M, EXAMPLE_TILE_N>,
                     pto::Stride<EXAMPLE_TOTAL_M * EXAMPLE_TILE_N, EXAMPLE_TOTAL_M * EXAMPLE_TILE_N,
                                 EXAMPLE_TOTAL_M * EXAMPLE_TILE_N, EXAMPLE_TILE_N, 1>>;

    using TileMatA = Tile<TileType::Mat, ExampleInT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, EXAMPLE_TOTAL_M,
                          EXAMPLE_TILE_K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, ExampleInT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, EXAMPLE_TILE_K,
                          EXAMPLE_TILE_N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<ExampleInT, ALIGNED_M, ALIGNED_K, EXAMPLE_TOTAL_M, EXAMPLE_TILE_K>;
    using RightTile = TileRight<ExampleInT, ALIGNED_K, ALIGNED_N, EXAMPLE_TILE_K, EXAMPLE_TILE_N>;
    using AccTile = TileAcc<ExampleOutT, EXAMPLE_TOTAL_M, EXAMPLE_TILE_N, EXAMPLE_TOTAL_M, EXAMPLE_TILE_N>;

    if constexpr (DAV_CUBE) {
        TileMatA aMatTile;
        TileMatB bMatTile;
        LeftTile aTile;
        RightTile bTile;
        AccTile accTile;
        TASSIGN(aMatTile, 0x0);
        TASSIGN(bMatTile, 0x20000);
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        GlobalA globalA(srcA);
        GlobalB globalB(srcB);
        GlobalOut globalOut(out);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        TLOAD(aMatTile, globalA);
        TLOAD(bMatTile, globalB);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        TMATMUL(accTile, aTile, bTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        TSTORE<AccTile, GlobalOut>(globalOut, accTile);

        pipe_barrier(PIPE_ALL);
    }
}
#else
__global__ AICORE void runTPushPopMatmulAdd(__gm__ uint64_t *ffts_addr, __gm__ ExampleOutT *out,
                                            __gm__ ExampleInT *srcA, __gm__ ExampleInT *srcB,
                                            __gm__ ExampleOutT *bias, __gm__ ExampleOutT *fifoMem)
{
    set_ffts_base_addr((uint64_t)ffts_addr);
    constexpr uint32_t NUM_M_TILES = EXAMPLE_TOTAL_M / EXAMPLE_CASE_TILE_M;
    constexpr uint32_t VEC_M = EXAMPLE_CASE_TILE_M / VEC_CORES;

    constexpr uint16_t FLAG_ID = 0;
    constexpr uint8_t FIFO_DEPTH = 2;
    constexpr uint8_t FIFO_PERIOD = 1;
    // local fifo base used for TPOP of vector side(vecTileHalf)
    constexpr uint32_t localFiFoBase = 0x0;

    using AccTile = TileAcc<ExampleOutT, EXAMPLE_CASE_TILE_M, EXAMPLE_TILE_N, EXAMPLE_CASE_TILE_M, EXAMPLE_TILE_N>;
    using VecTileHalf =
        Tile<TileType::Vec, ExampleOutT, VEC_M, EXAMPLE_TILE_N, BLayout::RowMajor, VEC_M, EXAMPLE_TILE_N>;
    using BiasTile =
        Tile<TileType::Vec, ExampleOutT, VEC_M, EXAMPLE_TILE_N, BLayout::RowMajor, VEC_M, EXAMPLE_TILE_N>;
    using OutTile =
        Tile<TileType::Vec, ExampleOutT, VEC_M, EXAMPLE_TILE_N, BLayout::RowMajor, VEC_M, EXAMPLE_TILE_N>;

    using MatPipe = TPipe<FLAG_ID, Direction::DIR_C2V,
                          EXAMPLE_CASE_TILE_M * EXAMPLE_TILE_N * sizeof(ExampleOutT), FIFO_DEPTH>;
    MatPipe mPipe((__gm__ void *)(uint64_t)fifoMem, 0x0, localFiFoBase);

    constexpr uint32_t blockAlign = C0_SIZE_BYTE / sizeof(ExampleInT);
    constexpr uint32_t ALIGNED_M = CeilAlign<uint32_t>(EXAMPLE_CASE_TILE_M, 16);
    constexpr uint32_t ALIGNED_K = CeilAlign<uint32_t>(EXAMPLE_TILE_K, blockAlign);
    constexpr uint32_t ALIGNED_N = CeilAlign<uint32_t>(EXAMPLE_TILE_N, blockAlign);

    using GlobalA =
        GlobalTensor<ExampleInT, pto::Shape<1, 1, 1, EXAMPLE_CASE_TILE_M, EXAMPLE_TILE_K>,
                     pto::Stride<EXAMPLE_TOTAL_M * EXAMPLE_TILE_K, EXAMPLE_TOTAL_M * EXAMPLE_TILE_K,
                                 EXAMPLE_CASE_TILE_M * EXAMPLE_TILE_K, EXAMPLE_TILE_K, 1>>;
    using GlobalB =
        GlobalTensor<ExampleInT, pto::Shape<1, 1, 1, EXAMPLE_TILE_K, EXAMPLE_TILE_N>,
                     pto::Stride<EXAMPLE_TILE_K * EXAMPLE_TILE_N, EXAMPLE_TILE_K * EXAMPLE_TILE_N,
                                 EXAMPLE_TILE_K * EXAMPLE_TILE_N, EXAMPLE_TILE_N, 1>>;
    using GlobalBias =
        GlobalTensor<ExampleOutT, pto::Shape<1, 1, 1, VEC_M, EXAMPLE_TILE_N>,
                     pto::Stride<EXAMPLE_TOTAL_M * EXAMPLE_TILE_N, EXAMPLE_TOTAL_M * EXAMPLE_TILE_N,
                                 VEC_M * EXAMPLE_TILE_N, EXAMPLE_TILE_N, 1>>;
    using GlobalOut =
        GlobalTensor<ExampleOutT, pto::Shape<1, 1, 1, VEC_M, EXAMPLE_TILE_N>,
                     pto::Stride<EXAMPLE_TOTAL_M * EXAMPLE_TILE_N, EXAMPLE_TOTAL_M * EXAMPLE_TILE_N,
                                 VEC_M * EXAMPLE_TILE_N, EXAMPLE_TILE_N, 1>>;

    using TileMatA = Tile<TileType::Mat, ExampleInT, ALIGNED_M, ALIGNED_K, BLayout::ColMajor, EXAMPLE_CASE_TILE_M,
                          EXAMPLE_TILE_K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, ExampleInT, ALIGNED_K, ALIGNED_N, BLayout::ColMajor, EXAMPLE_TILE_K,
                          EXAMPLE_TILE_N, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<ExampleInT, ALIGNED_M, ALIGNED_K, EXAMPLE_CASE_TILE_M, EXAMPLE_TILE_K>;
    using RightTile = TileRight<ExampleInT, ALIGNED_K, ALIGNED_N, EXAMPLE_TILE_K, EXAMPLE_TILE_N>;

    if constexpr (DAV_CUBE) {
        TileMatA aMatTile;
        TileMatB bMatTile;
        TASSIGN(aMatTile, 0x0);
        TASSIGN(bMatTile, 0x20000);

        LeftTile aTile;
        RightTile bTile;
        AccTile accTile;
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

        for (int m_tile = 0; m_tile < NUM_M_TILES; m_tile++) {
            GlobalA globalA(srcA + m_tile * EXAMPLE_CASE_TILE_M * EXAMPLE_TILE_K);
            GlobalB globalB(srcB);

            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

            TLOAD(aMatTile, globalA);
            TLOAD(bMatTile, globalB);

            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

            TMOV(aTile, aMatTile);
            TMOV(bTile, bMatTile);

            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);

            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

            TMATMUL(accTile, aTile, bTile);

            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

            TPUSH<MatPipe, AccTile, TileSplitAxis::TILE_UP_DOWN>(mPipe, accTile);

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        }

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }

    if constexpr (DAV_VEC) {
        VecTileHalf vecTileHalf;
        BiasTile biasTile;
        OutTile outTile;
        TASSIGN(biasTile, 0x10000);
        TASSIGN(outTile, 0x20000);

        uint32_t subBlockIdx = get_subblockid();

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        for (int m_tile = 0; m_tile < NUM_M_TILES; m_tile++) {
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

            TPOP<MatPipe, VecTileHalf, TileSplitAxis::TILE_UP_DOWN>(mPipe, vecTileHalf);

            size_t biasOffset =
                static_cast<size_t>(m_tile * EXAMPLE_CASE_TILE_M + subBlockIdx * VEC_M) * EXAMPLE_TILE_N;
            GlobalBias globalBias(bias + biasOffset);

            TLOAD(biasTile, globalBias);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

            TADD(outTile, vecTileHalf, biasTile);

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            size_t outOffset =
                static_cast<size_t>(m_tile * EXAMPLE_CASE_TILE_M + subBlockIdx * VEC_M) * EXAMPLE_TILE_N;
            GlobalOut globalOut(out + outOffset);
            TSTORE(globalOut, outTile);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

        pipe_barrier(PIPE_ALL);
    }
}
#endif

void LaunchTPushPopMatmulAdd(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *bias, uint8_t *fifoMem,
                             void *stream)
{
#ifdef TPUSHPOP_SANITY_ONLY
    (void)ffts;
    (void)bias;
    (void)fifoMem;
    runSanityMatmul<<<1, nullptr, stream>>>(
        reinterpret_cast<ExampleOutT *>(out), reinterpret_cast<ExampleInT *>(srcA), reinterpret_cast<ExampleInT *>(srcB));
#else
    runTPushPopMatmulAdd<<<1, nullptr, stream>>>(
        reinterpret_cast<uint64_t *>(ffts), reinterpret_cast<ExampleOutT *>(out), reinterpret_cast<ExampleInT *>(srcA),
        reinterpret_cast<ExampleInT *>(srcB), reinterpret_cast<ExampleOutT *>(bias), reinterpret_cast<ExampleOutT *>(fifoMem));
#endif
}
