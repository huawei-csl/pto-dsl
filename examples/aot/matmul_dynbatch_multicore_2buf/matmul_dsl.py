from mlir.ir import IntegerType

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s


def build(M=128, K=128, N=128):
    def meta_data():
        dtype = pto.float16
        dtype_acc_tile = pto.float32
        ptr_type = pto.PtrType(dtype)
        i32 = pto.int32
        i1 = IntegerType.get_signless(1)

        tensor_type = pto.TensorType(rank=2, dtype=dtype)
        tensor_type3d = pto.TensorType(rank=3, dtype=dtype)

        tile_view_a = pto.SubTensorType(shape=[M, K], dtype=dtype)
        tile_view_b = pto.SubTensorType(shape=[K, N], dtype=dtype)
        tile_view_c = pto.SubTensorType(shape=[M, N], dtype=dtype)
        tile_buf_aMat = pto.TileBufType(shape=[M, K], dtype=dtype, memory_space="MAT")
        tile_buf_bMat = pto.TileBufType(shape=[K, N], dtype=dtype, memory_space="MAT")
        tile_buf_aTile = pto.TileBufType(shape=[M, K], dtype=dtype, memory_space="LEFT")
        tile_buf_bTile = pto.TileBufType(shape=[K, N], dtype=dtype, memory_space="RIGHT")
        tile_buf_cTile = pto.TileBufType(shape=[M, N], dtype=dtype_acc_tile, memory_space="ACC")
        # TODO: Get rid of this?
        return locals()

    const = s.const

    @to_ir_module(meta_data=meta_data)
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
            c2 = const(2)
            cM = const(M)
            cK = const(K)
            cN = const(N)
            batch = s.index_cast(batch_i32)

            num_blocks = s.index_cast(pto.get_block_num())
            # TODO round robin
            batches_per_core = s.ceil_div(batch, num_blocks)
            bid = s.index_cast(pto.get_block_idx())
            b_start = bid * batches_per_core
            b_end_unclamped = b_start + batches_per_core
            b_end = s.min_u(b_end_unclamped, batch)

            # TODO: if no batched assigned to this core, early return

            tvA = pto.as_tensor(tensor_type3d, ptr=a_ptr, shape=[batch, cM, cK], strides=[cK*cM, cK, c1])
            tvC = pto.as_tensor(tensor_type3d, ptr=out_ptr, shape=[batch, cM, cN], strides=[cM*cN, cN, c1])
            tvB = pto.as_tensor(tensor_type, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])

            # TODO: pre-fetch more than two tiles into L1
            NUM_BUFFERS = 2
            aMatTiles = [pto.alloc_tile(tile_buf_aMat) for _ in range(NUM_BUFFERS)]
            bMatTile = pto.alloc_tile(tile_buf_bMat)
            # Ping and pong buffers in L0A/C
            aTiles = [pto.alloc_tile(tile_buf_aTile) for _ in range(NUM_BUFFERS)]
            cTiles = [pto.alloc_tile(tile_buf_cTile) for _ in range(NUM_BUFFERS)]
            bTile = pto.alloc_tile(tile_buf_bTile)

            # Put B in L0B
            svB = pto.slice_view(tile_view_b, source=tvB, offsets=[c0, c0], sizes=[cK, cN])
            pto.load(svB, bMatTile)
            tile.mov(bMatTile, bTile)
            # TODO: wait here so we can use full l1 memory later for A.


            # load in the first tile from GM->L1
            svA = pto.slice_view(tile_view_a, source=tvA, offsets=[b_start, c0, c0], sizes=[c1, cM, cK])
            curr = c1 - (b_start % c2)
            pto.cond(
                curr == c1,
                lambda: pto.load(svA, aMatTiles[0]),
                lambda: pto.load(svA, aMatTiles[1]),
            )

            for b_idx in pto.range(b_start, b_end, c1):
                curr = b_idx % c2
                svA = pto.slice_view(tile_view_a, source=tvA, offsets=[b_idx+c1, c0, c0], sizes=[c1, cM, cK])
                svC = pto.slice_view(tile_view_c, source=tvC, offsets=[b_idx, c0, c0], sizes=[c1, cM, cN])

                ########## Load tile A for iteration i+1 from GM -> L1
                with pto.if_context(b_idx + c1 < b_end):
                    pto.cond(
                        curr == c1,
                        lambda: pto.load(svA, aMatTiles[0]),
                        lambda: pto.load(svA, aMatTiles[1])
                    )


                ########## Move A1 and A2 into L0A
                pto.cond(
                    curr == c0,
                    lambda: tile.mov(aMatTiles[0], aTiles[0]),
                    lambda: tile.mov(aMatTiles[1], aTiles[1])
                )


                ########## Perform matmul
                pto.cond(
                    curr == c0,
                    lambda: tile.matmul(aTiles[0], bTile, cTiles[0]),
                    lambda: tile.matmul(aTiles[1], bTile, cTiles[1]),
                )


                ######### Store
                pto.cond(
                    curr == c0,
                    lambda: pto.store(cTiles[0], svC),
                    lambda: pto.store(cTiles[1], svC),
                )
                pto.barrier('LOAD')


    return RunTMATMULSplitK


if __name__ == "__main__":
    print(build())
