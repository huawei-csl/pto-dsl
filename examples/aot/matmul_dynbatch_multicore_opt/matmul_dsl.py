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
        tile_view_c = pto.SubTensorType(shape=[M, N], dtype=dtype)
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
            "tile_view_c": tile_view_c,
            "tile_buf_aMat": tile_buf_aMat,
            "tile_buf_bMat": tile_buf_bMat,
            "tile_buf_aTile": tile_buf_aTile,
            "tile_buf_bTile": tile_buf_bTile,
            "tile_buf_cTile": tile_buf_cTile,
        }

    const = pto.const


    # Until we have set_dyn_flag with event_id as SSA values
    # event_id can be dynamic SSA value
    # https://github.com/zhangstevenunity/PTOAS/pull/176
    def record_event(src, dst, event_id):
        pto.cond(
            event_id == const(0),
            lambda: pto.record_event(src, dst, event_id=0),
            lambda: pto.record_event(src, dst, event_id=1)
        )
     
    def wait_event(src, dst, event_id):
        pto.cond(
            event_id == const(0),
            lambda: pto.wait_event(src, dst, event_id=0),
            lambda: pto.wait_event(src, dst, event_id=1)
        )   

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
            batch = pto.index_cast(batch_i32)

            num_blocks = pto.index_cast(pto.get_block_num())
            # TODO round robin
            batches_per_core = pto.ceil_div(batch, num_blocks)
            bid = pto.index_cast(pto.get_block_idx())
            b_start = bid * batches_per_core
            b_end_unclamped = b_start + batches_per_core
            b_end = pto.min_u(b_end_unclamped, batch)

            # TODO: if no batched assigned to this core, do nothing...

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
            pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)
            pto.mov(bMatTile, bTile)
            # TODO: wait here so we can use full l1 memory later for A.


            # load in the first tile from GM->L1
            svA = pto.slice_view(tile_view_a, source=tvA, offsets=[b_start, c0, c0], sizes=[c1, cM, cK])
            curr = c1 - (b_start % c2)
            pto.cond(
                curr == c1,
                lambda: pto.load(svA, aMatTiles[0]),
                lambda: pto.load(svA, aMatTiles[1]),
            )
            record_event("LOAD", "MOV_M2L", event_id=curr)

            # signal to LOAD that L1 can be overwritten
            pto.record_event("MOV_M2L", "LOAD", event_id=[0, 1])
            # signal to MOV that L0 can be overwritten
            pto.record_event("MATMUL", "MOV_M2L", event_id=[0, 1])
            # signal to MATMUL that it can overwrite L0C
            pto.record_event("STORE_ACC", "MATMUL", event_id=[0, 1])

            for b_idx in pto.for_range(b_start, b_end, c1):
                curr = b_idx % c2
                svA = pto.slice_view(tile_view_a, source=tvA, offsets=[b_idx+c1, c0, c0], sizes=[c1, cM, cK])
                svC = pto.slice_view(tile_view_c, source=tvC, offsets=[b_idx, c0, c0], sizes=[c1, cM, cN])

                ########## Load tile A for iteration i+1 from GM -> L1
                wait_event("MOV_M2L", "LOAD", event_id=curr)
                with pto.if_context(b_idx + c1 < b_end):
                    pto.cond(
                        curr == c1,
                        lambda: pto.load(svA, aMatTiles[0]),
                        lambda: pto.load(svA, aMatTiles[1])
                    )
                    record_event("LOAD", "MOV_M2L", event_id=curr)


                ########## Move A1 and A2 into L0A
                wait_event("LOAD", "MOV_M2L", event_id=c1 - curr)
                wait_event("MATMUL", "MOV_M2L", event_id=curr)
                pto.cond(
                    curr == c0,
                    lambda: pto.mov(aMatTiles[0], aTiles[0]),
                    lambda: pto.mov(aMatTiles[1], aTiles[1])
                )
                with pto.if_context(b_idx + c2 < b_end):
                    record_event("MOV_M2L", "LOAD", event_id=curr)
                record_event("MOV_M2L", "MATMUL", event_id=curr)


                ########## Perform matmul
                wait_event("MOV_M2L", "MATMUL", event_id=curr)
                wait_event("STORE_ACC", "MATMUL", event_id=curr)
                pto.cond(
                    curr == c0,
                    lambda: pto.matmul(aTiles[0], bTile, cTiles[0]),
                    lambda: pto.matmul(aTiles[1], bTile, cTiles[1]),
                )
                record_event("MATMUL", "STORE_ACC", event_id=curr)
                with pto.if_context(b_idx + c2 < b_end):
                    record_event("MATMUL", "MOV_M2L", event_id=curr)


                ######### Store
                wait_event("MATMUL", "STORE_ACC", event_id=curr)
                pto.cond(
                    curr == c0,
                    lambda: pto.store(cTiles[0], svC),
                    lambda: pto.store(cTiles[1], svC),
                )
                with pto.if_context(b_idx + c2 < b_end):
                    record_event("STORE_ACC", "MATMUL", event_id=curr)


    return RunTMATMULSplitK


if __name__ == "__main__":
    print(build())
