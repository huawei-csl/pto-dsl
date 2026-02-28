# adapted from https://github.com/zhangstevenunity/PTOAS/blob/a301aa43b388d9b2e1ba0db8773b3a719e8c445b/test/samples/MatMul/tmatmulk.py

from ptodsl import to_ir_module
import ptodsl.language as pto


def build(
    M=128,
    K=128,
    N=128,
    validM=128,
    validK=128,
    validN=128,
):
    def meta_data():
        dtype = pto.float32
        ptr_type = pto.PtrType(dtype)
        bool_type = pto.bool
        index_dtype = pto.int32

        tv_a = pto.TensorType(rank=3, dtype=dtype)
        tv_b = pto.TensorType(rank=2, dtype=dtype)
        tv_out = pto.TensorType(rank=3, dtype=dtype)

        tile_view_a = pto.SubTensorType(shape=[M, K], dtype=dtype)
        tile_view_b = pto.SubTensorType(shape=[K, N], dtype=dtype)
        tile_view_out = pto.SubTensorType(shape=[M, N], dtype=dtype)

        tile_buf_aMat = pto.TileBufType(shape=[M, K], dtype=dtype, memory_space="MAT")
        tile_buf_bMat = pto.TileBufType(shape=[K, N], dtype=dtype, memory_space="MAT")
        tile_buf_aTile = pto.TileBufType(shape=[M, K], dtype=dtype, memory_space="LEFT")
        tile_buf_bTile = pto.TileBufType(shape=[K, N], dtype=dtype, memory_space="RIGHT")
        tile_buf_cTile = pto.TileBufType(shape=[M, N], dtype=dtype, memory_space="ACC")

        return {
            "ptr_type": ptr_type,
            "bool_type": bool_type,
            "index_dtype": index_dtype,
            "tv_a": tv_a,
            "tv_b": tv_b,
            "tv_out": tv_out,
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
        isBias: "bool_type",
        batch_i32: "index_dtype",
    ) -> None:
        # Keep unused args to preserve original function signature/ABI.
        _ = bias_ptr
        _ = isBias

        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            cM = const(validM)
            cK = const(validK)
            cN = const(validN)
            cKM = const(validK * validM)
            cMN = const(validM * validN)
            cTileM = const(M)
            cTileN = const(N)

            batch = pto.index_cast(batch_i32)

            # Distribute batches over cores with "base + remainder" policy.
            num_blocks = pto.index_cast(pto.get_block_num())
            bid = pto.index_cast(pto.get_block_idx())

            base = pto.div_s(batch, num_blocks)
            rem = pto.rem_s(batch, num_blocks)
            lt_rem = pto.lt(bid, rem)
            min_bid_rem = pto.min_u(bid, rem)
            b_start = bid * base + min_bid_rem
            length = base + pto.select(lt_rem, c1, c0)
            b_end = pto.min_u(b_start + length, batch)

            tvA = pto.as_tensor(tv_a, ptr=a_ptr, shape=[batch, cM, cK], strides=[cKM, cK, c1])
            tvB = pto.as_tensor(tv_b, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
            tvOut = pto.as_tensor(tv_out, ptr=out_ptr, shape=[batch, cM, cN], strides=[cMN, cN, c1])

            aMatTile = pto.alloc_tile(tile_buf_aMat)
            bMatTile = pto.alloc_tile(tile_buf_bMat)
            aTile = pto.alloc_tile(tile_buf_aTile)
            bTile = pto.alloc_tile(tile_buf_bTile)
            cTile = pto.alloc_tile(tile_buf_cTile)

            # B is shared across batches: load once GM->L1->L0B.
            svB = pto.slice_view(tile_view_b, source=tvB, offsets=[c0, c0], sizes=[cK, cTileN])
            pto.load(svB, bMatTile)
            pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)
            pto.mov(bMatTile, bTile)

            for b_idx in pto.for_range(b_start, b_end, c1):
                svA = pto.slice_view(
                    tile_view_a,
                    source=tvA,
                    offsets=[b_idx, c0, c0],
                    sizes=[c1, cTileM, cK],
                )
                svOut = pto.slice_view(
                    tile_view_out,
                    source=tvOut,
                    offsets=[b_idx, c0, c0],
                    sizes=[c1, cTileM, cTileN],
                )

                pto.load(svA, aMatTile)
                pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)

                pto.mov(aMatTile, aTile)
                pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
                pto.matmul(aTile, bTile, cTile)
                pto.record_wait_pair("MATMUL", "LOAD", event_id=0)

                pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
                pto.store(cTile, svOut)
                pto.record_wait_pair("STORE_ACC", "MATMUL", event_id=0)

    return RunTMATMULSplitK


if __name__ == "__main__":
    m = build()
    print(m)