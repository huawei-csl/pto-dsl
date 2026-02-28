# adapted from https://github.com/zhangstevenunity/PTOAS/blob/a301aa43b388d9b2e1ba0db8773b3a719e8c445b/test/samples/MatMul/tmatmulk.py

from ptodsl import to_ir_module
import ptodsl.language as pto


def build(
    M=32,
    K=256,
    N=32,
    validM=32,
    validK=256,
    validN=32,
    BASEK=32,
):
    assert K % BASEK == 0
    iters = K // BASEK

    def meta_data():
        dtype = pto.float32
        i1 = pto.bool
        ptr_type = pto.PtrType(dtype)

        tensor_type = pto.TensorType(rank=2, dtype=dtype)

        tile_view_a = pto.SubTensorType(shape=[M, BASEK], dtype=dtype)
        tile_view_b = pto.SubTensorType(shape=[BASEK, N], dtype=dtype)
        tile_view_out = pto.SubTensorType(shape=[M, N], dtype=dtype)
        tile_view_bias = pto.SubTensorType(shape=[1, N], dtype=dtype)

        tile_buf_aMat = pto.TileBufType(shape=[M, BASEK], dtype=dtype, memory_space="MAT")
        tile_buf_bMat = pto.TileBufType(shape=[BASEK, N], dtype=dtype, memory_space="MAT")
        tile_buf_biasData = pto.TileBufType(shape=[1, N], dtype=dtype, memory_space="MAT")
        tile_buf_aTile = pto.TileBufType(shape=[M, BASEK], dtype=dtype, memory_space="LEFT")
        tile_buf_bTile = pto.TileBufType(shape=[BASEK, N], dtype=dtype, memory_space="RIGHT")
        tile_buf_cTile = pto.TileBufType(shape=[M, N], dtype=dtype, memory_space="ACC")
        tile_buf_biasTile = pto.TileBufType(
            shape=[1, N], dtype=dtype, memory_space="BIAS"
        )

        return {
            "ptr_type": ptr_type,
            "i1": i1,
            "tensor_type": tensor_type,
            "tile_view_a": tile_view_a,
            "tile_view_b": tile_view_b,
            "tile_view_out": tile_view_out,
            "tile_view_bias": tile_view_bias,
            "tile_buf_aMat": tile_buf_aMat,
            "tile_buf_bMat": tile_buf_bMat,
            "tile_buf_biasData": tile_buf_biasData,
            "tile_buf_aTile": tile_buf_aTile,
            "tile_buf_bTile": tile_buf_bTile,
            "tile_buf_cTile": tile_buf_cTile,
            "tile_buf_biasTile": tile_buf_biasTile,
        }

    const = pto.const

    @to_ir_module(meta_data=meta_data)
    def RunTMATMULSplitK(
        out_ptr: "ptr_type",
        a_ptr: "ptr_type",
        b_ptr: "ptr_type",
        bias_ptr: "ptr_type",
        isBias: "i1",
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

            tvA = pto.as_tensor(tensor_type, ptr=a_ptr, shape=[cM, cK], strides=[cK, c1])
            tvB = pto.as_tensor(tensor_type, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
            tvOut = pto.as_tensor(tensor_type, ptr=out_ptr, shape=[cM, cN], strides=[cN, c1])
            tvBias = pto.as_tensor(tensor_type, ptr=bias_ptr, shape=[c1, cN], strides=[cN, c1])

            aMatTile = pto.alloc_tile(tile_buf_aMat)
            bMatTile = pto.alloc_tile(tile_buf_bMat)
            biasDataTile = pto.alloc_tile(tile_buf_biasData)
            aTile = pto.alloc_tile(tile_buf_aTile)
            bTile = pto.alloc_tile(tile_buf_bTile)
            cTile = pto.alloc_tile(tile_buf_cTile)
            biasTile = pto.alloc_tile(tile_buf_biasTile)

            for i in pto.for_range(c0, cIter, c1):
                kOff = i * cBASEK
                svA = pto.slice_view(
                    tile_view_a,
                    source=tvA,
                    offsets=[c0, kOff],
                    sizes=[cTileM, cBASEK],
                )
                svB = pto.slice_view(
                    tile_view_b,
                    source=tvB,
                    offsets=[kOff, c0],
                    sizes=[cBASEK, cTileN],
                )
                svBias = pto.slice_view(
                    tile_view_bias,
                    source=tvBias,
                    offsets=[c0, c0],
                    sizes=[c1, cTileN],
                )

                pto.load(svA, aMatTile)
                pto.load(svB, bMatTile)
                with pto.if_context(isBias):
                    pto.load(svBias, biasDataTile)

                pto.mov(aMatTile, aTile)
                pto.mov(bMatTile, bTile)
                with pto.if_context(isBias):
                    pto.mov(biasDataTile, biasTile)

                is_i0 = pto.eq(i, c0)

                def _first_iter():
                    pto.cond(
                        isBias,
                        lambda: pto.matmul_bias(aTile, bTile, biasTile, cTile),
                        lambda: pto.matmul(aTile, bTile, cTile),
                    )

                pto.cond(
                    is_i0,
                    _first_iter,
                    lambda: pto.matmul_acc(cTile, aTile, bTile, cTile),
                )

            svOut = pto.slice_view(
                tile_view_out,
                source=tvOut,
                offsets=[c0, c0],
                sizes=[cTileM, cTileN],
            )
            pto.store(cTile, svOut)

    module = RunTMATMULSplitK
    return module


if __name__ == "__main__":
    print(build())