from mlir.ir import IntegerType

from ptodsl import to_ir_module
import ptodsl.language as pto


def build(M=128, K=128, N=128):
    def meta_data():
        dtype = pto.float16
        dtype_acc_tile = pto.float32
        ptr_dtype = pto.PtrType(dtype)
        i32 = pto.int32
        i1 = IntegerType.get_signless(1)

        tensor_type = pto.TensorType(rank=2, dtype=dtype)
        tensor_type3d = pto.TensorType(rank=3, dtype=dtype)

        tile_view_a = pto.SubTensorType(shape=[M, K], dtype=dtype)
        tile_view_b = pto.SubTensorType(shape=[K, N], dtype=dtype)
        tile_view_out = pto.SubTensorType(shape=[M, N], dtype=dtype)
        tile_buf_aMat = pto.TileBufType(shape=[M, K], dtype=dtype, memory_space="MAT")
        tile_buf_bMat = pto.TileBufType(shape=[K, N], dtype=dtype, memory_space="MAT")
        tile_buf_aTile = pto.TileBufType(shape=[M, K], dtype=dtype, memory_space="LEFT")
        tile_buf_bTile = pto.TileBufType(shape=[K, N], dtype=dtype, memory_space="RIGHT")
        tile_buf_cTile = pto.TileBufType(shape=[M, N], dtype=dtype_acc_tile, memory_space="ACC")

        return {
            "ptr_type": ptr_dtype,
            "i32": i32,
            "i1": i1,
            "tensor_type": tensor_type,
            "tensor_type3d": tensor_type3d,
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
        bias_ptr: "ptr_type",
        isBias: "i1",
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

            tvA = pto.as_tensor(tensor_type3d, ptr=a_ptr, shape=[batch, cM, cK], strides=[cK*CM, cK, c1])
            tvB = pto.as_tensor(tensor_type, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
            tvOut = pto.as_tensor(tensor_type3d, ptr=out_ptr, shape=[batch, cM, cN], strides=[cM*cN, cN, c1])

            aMatTile = pto.alloc_tile(tile_buf_aMat)
            bMatTile = pto.alloc_tile(tile_buf_bMat)
            aTile = pto.alloc_tile(tile_buf_aTile)
            bTile = pto.alloc_tile(tile_buf_bTile)
            cTile = pto.alloc_tile(tile_buf_cTile)

            # signal to LOAD that L1 can be overwritten
            pto.record_event("MOV_M2L", "LOAD", event_id=0)
            # signal to MOV that L0 can be overwritten
            pto.record_event("MATMUL", "MOV_M2L", event_id=0)
            # signal to MATMUL that it can overwrite L0C
            pto.record_event("STORE_ACC", "MATMUL", event_id=0)
            for b_idx in pto.for_range(b_start, b_end, c1):
                row_off = b_idx * cM

                svA = pto.slice_view(tile_view_a, source=tvA, offsets=[b_idx, c0, c0], sizes=[c1, cM, cK])
                svB = pto.slice_view(tile_view_b, source=tvB, offsets=[c0, c0], sizes=[cK, cN])
                svOut = pto.slice_view(tile_view_out, source=tvOut, offsets=[b_idx, c0, c0], sizes=[c1, cM, cN])

                pto.wait_event("MOV_M2L", "LOAD", event_id=0)
                pto.load(svA, aMatTile)
                pto.load(svB, bMatTile)

                # Before moving data from L1 into L0 we must know
                # 1) The load has finished
                # 2) the L0 can be overwritten
                pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)
                pto.wait_event("MATMUL", "MOV_M2l", event_id=0)

                pto.mov(aMatTile, aTile)
                pto.mov(bMatTile, bTile)
                # signal to LOAD that L1 can be overwritten
                pto.record_event("MOV_M2L", "LOAD", event_id=0)

                # Moves into L0 must finish before we do the matmul
                pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
                # L0C must be ready to be overwritten
                pto.wait_event("STORE_ACC", "MATMUL", event_id=0)
                pto.matmul(aTile, bTile, cTile)

                # matmul must be completed before we do store
                pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
                # also signal to MOV that L0A/B can be overwritten again
                pto.record_event("MATMUL", "MOV_M2L", event_id=0)
                pto.store(cTile, svOut)

                # signal to MATMUL that it can overwrite L0C
                pto.record_event("STORE_ACC", "MATMUL", event_id=0)

            pto.wait_event("MOV_M2L", "LOAD", event_id=0)
            pto.wait_event("MATMUL", "MOV_M2L", event_id=0)
            pto.wait_event("STORE_ACC", "MATMUL", event_id=0)

    return RunTMATMULSplitK


if __name__ == "__main__":
    print(build())