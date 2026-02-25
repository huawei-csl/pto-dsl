from mlir.ir import IntegerType

from ptodsl import to_ir_module
import ptodsl.language as pto


def build(
    M=128,
    K=128,
    N=128,
):
    def meta_data():
        dtype = pto.float32
        ptr_dtype = pto.PtrType(dtype)
        i32 = pto.int32

        # todo: add 3d tensors
        tensor_type = pto.TensorType(rank=2, dtype=dtype)

        tile_view_a = pto.SubTensorType(shape=[M, K], dtype=dtype)
        tile_view_b = pto.SubTensorType(shape=[K, N], dtype=dtype)
        tile_view_out = pto.SubTensorType(shape=[M, N], dtype=dtype)

        tile_buf_aMat = pto.TileBufType(shape=[M, K], dtype=dtype, memory_space="MAT")
        tile_buf_bMat = pto.TileBufType(shape=[K, N], dtype=dtype, memory_space="MAT")

        tile_buf_aTile = pto.TileBufType(shape=[M, K], dtype=dtype, memory_space="LEFT")
        tile_buf_bTile = pto.TileBufType(shape=[K, N], dtype=dtype, memory_space="RIGHT")
        tile_buf_cTile = pto.TileBufType(shape=[M, N], dtype=dtype, memory_space="ACC")

        return {
            "ptr_type": ptr_dtype,
            "i32": i32,
            "tensor_type": tensor_type,
            "tile_view_a": tile_view_a,
            "tile_view_b": tile_view_b,
            "tile_view_out": tile_view_out,
            "tile_buf_aMat": tile_buf_aMat,
            "tile_buf_bMat": tile_buf_bMat,
            "tile_buf_aTile": tile_buf_aTile,
            "tile_buf_bTile": tile_buf_bTile,
            "tile_buf_cTile": tile_buf_cTile,
        }

    const = pto.const

    @to_ir_module(meta_data=meta_data)
    def RunTMATMULSplitK(
        out_ptr: "ptr_type",
        a_ptr: "ptr_type",
        b_ptr: "ptr_type",
        batch_i32: "i32",
    ) -> None:
        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            cM = const(M)
            cK = const(K)
            cN = const(N)

            batch = pto.index_cast(batch_i32)
            cBM = batch * cM

            num_blocks = pto.index_cast(pto.get_block_num())
            batches_per_core = pto.ceil_div(batch, num_blocks)
            bid = pto.index_cast(pto.get_block_idx())
            b_start = bid * batches_per_core
            b_end_unclamped = b_start + batches_per_core
            b_end = pto.min_u(b_end_unclamped, batch)

            tvA = pto.as_tensor(tensor_type, ptr=a_ptr, shape=[cBM, cK], strides=[cK, c1])
            tvB = pto.as_tensor(tensor_type, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
            tvOut = pto.as_tensor(tensor_type, ptr=out_ptr, shape=[cBM, cN], strides=[cN, c1])

            aMatTile = pto.alloc_tile(tile_buf_aMat)
            bMatTile = pto.alloc_tile(tile_buf_bMat)
            aTile = pto.alloc_tile(tile_buf_aTile)
            bTile = pto.alloc_tile(tile_buf_bTile)
            cTile = pto.alloc_tile(tile_buf_cTile)

            for b_idx in pto.for_range(b_start, b_end, c1):
                row_off = b_idx * cM

                svA = pto.slice_view(tile_view_a, source=tvA, offsets=[row_off, c0], sizes=[cM, cK])
                svB = pto.slice_view(tile_view_b, source=tvB, offsets=[c0, c0], sizes=[cK, cN])
                svBias = pto.slice_view(tile_view_bias, source=tvBias, offsets=[c0, c0], sizes=[c1, cN])

                pto.load(svA, aMatTile)
                pto.load(svB, bMatTile)

                # Load from GM->L1 must finish before we move into L0
                pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)

                pto.mov(aMatTile, aTile)
                pto.mov(bMatTile, bTile)

                # Moves into L0 must finish before we do the matmul
                pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
                pto.matmul(aTile, bTile, cTile),
                
                # not needed since no loop again so we wont be loading?
                # matmul must be completed before we load?
                # Not really? when doing matmul we must make sure L0 is not overwritten
                # Load can happen at same time?
                pto.record_wait_pair("MATMUL", "LOAD", event_id=0)

                # matmul must be completed before we do store
                pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
                svOut = pto.slice_view(tile_view_out, source=tvOut, offsets=[row_off, c0], sizes=[cM, cN])
                pto.store(cTile, svOut)

                # Store must have finished writing back to GM before matmul comes in and overwrites the L0C
                pto.record_wait_pair("STORE_ACC", "MATMUL", event_id=0)

    return RunTMATMULSplitK


if __name__ == "__main__":
    print(build())