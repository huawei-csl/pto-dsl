from mlir.ir import IntegerType

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s


def build(
    M=128,
    K=128,
    N=128,
    validM=128,
    validK=128,
    validN=128,
    BASEK=32,
):
    assert K % BASEK == 0
    iters = K // BASEK

    dtype = pto.float32
    ptr_type = pto.PtrType(dtype)
    i1 = IntegerType.get_signless(1)
    i32 = pto.int32

    tensor_type = pto.TensorType(rank=2, dtype=dtype)

    tile_buf_aMat = pto.TileBufType(shape=[M, BASEK], dtype=dtype, memory_space="MAT")
    tile_buf_bMat = pto.TileBufType(shape=[BASEK, N], dtype=dtype, memory_space="MAT")
    tile_buf_biasData = pto.TileBufType(shape=[1, N], dtype=dtype, memory_space="MAT")

    tile_buf_aTile = pto.TileBufType(shape=[M, BASEK], dtype=dtype, memory_space="LEFT")
    tile_buf_bTile = pto.TileBufType(
        shape=[BASEK, N], dtype=dtype, memory_space="RIGHT"
    )
    tile_buf_cTile = pto.TileBufType(shape=[M, N], dtype=dtype, memory_space="ACC")
    tile_buf_biasTile = pto.TileBufType(shape=[1, N], dtype=dtype, memory_space="BIAS")

    const = s.const

    @to_ir_module
    def RunTMATMULSplitK(
        out_ptr: "ptr_type",
        a_ptr: "ptr_type",
        b_ptr: "ptr_type",
        bias_ptr: "ptr_type",
        isBias: "i1",
        batch_i32: "i32",
    ) -> None:
        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            cM = const(validM)
            cK = const(validK)
            cN = const(validN)
            cBASEK = const(BASEK)
            cIter = const(iters)
            cTileM = const(M)
            cTileN = const(N)

            batch = s.index_cast(batch_i32)
            cBM = batch * cM

            num_blocks = s.index_cast(pto.get_block_num())
            batches_per_core = s.ceil_div(batch, num_blocks)
            bid = s.index_cast(pto.get_block_idx())
            b_start = bid * batches_per_core
            b_end_unclamped = b_start + batches_per_core
            b_end = s.min_u(b_end_unclamped, batch)

            tvA = pto.as_tensor(ptr=a_ptr, shape=[cBM, cK], strides=[cK, c1])
            tvB = pto.as_tensor(ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
            tvOut = pto.as_tensor(ptr=out_ptr, shape=[cBM, cN], strides=[cN, c1])
            tvBias = pto.as_tensor(ptr=bias_ptr, shape=[c1, cN], strides=[cN, c1])

            aMatTile = pto.alloc_tile(tile_buf_aMat)
            bMatTile = pto.alloc_tile(tile_buf_bMat)
            biasDataTile = pto.alloc_tile(tile_buf_biasData)
            aTile = pto.alloc_tile(tile_buf_aTile)
            bTile = pto.alloc_tile(tile_buf_bTile)
            cTile = pto.alloc_tile(tile_buf_cTile)
            biasTile = pto.alloc_tile(tile_buf_biasTile)

            for b_idx in pto.range(b_start, b_end, c1):
                row_off = b_idx * cM

                for i in pto.range(c0, cIter, c1):
                    kOff = i * cBASEK
                    svA = pto.slice_view(
                        source=tvA,
                        offsets=[row_off, kOff],
                        sizes=[cTileM, cBASEK],
                    )
                    svB = pto.slice_view(
                        source=tvB,
                        offsets=[kOff, c0],
                        sizes=[cBASEK, cTileN],
                    )
                    svBias = pto.slice_view(
                        source=tvBias,
                        offsets=[c0, c0],
                        sizes=[c1, cTileN],
                    )

                    pto.load(svA, aMatTile)
                    pto.load(svB, bMatTile)
                    with pto.if_context(isBias):
                        pto.load(svBias, biasDataTile)

                    pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)

                    tile.mov(aMatTile, aTile)
                    tile.mov(bMatTile, bTile)
                    with pto.if_context(isBias):
                        tile.mov(biasDataTile, biasTile)

                    pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)

                    is_i0 = s.eq(i, c0)

                    def _first_iter():
                        pto.cond(
                            isBias,
                            lambda: tile.matmul_bias(aTile, bTile, biasTile, cTile),
                            lambda: tile.matmul(aTile, bTile, cTile),
                        )

                    pto.cond(
                        is_i0,
                        _first_iter,
                        lambda: tile.matmul_acc(cTile, aTile, bTile, cTile),
                    )

                    pto.record_wait_pair("MATMUL", "LOAD", event_id=0)

                pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
                svOut = pto.slice_view(
                    source=tvOut,
                    offsets=[row_off, c0],
                    sizes=[cTileM, cTileN],
                )
                pto.store(cTile, svOut)
                pto.record_wait_pair("STORE_ACC", "MATMUL", event_id=0)

    return RunTMATMULSplitK


if __name__ == "__main__":
    print(build())
