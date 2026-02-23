from mlir.ir import IntegerType

from ptodsl import jit
import ptodsl.language as pto
import torch
import torch_npu

const = pto.const


def build_kernel(
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

    def meta_data():
        dtype = pto.float32
        ptr_dtype = pto.PtrType(dtype)
        i1 = IntegerType.get_signless(1)
        i32 = pto.int32

        tv2_a = pto.TensorType(rank=2, dtype=dtype)
        tv2_b = pto.TensorType(rank=2, dtype=dtype)
        tv2_out = pto.TensorType(rank=2, dtype=dtype)
        tv2_bias = pto.TensorType(rank=2, dtype=dtype)

        tile_view_a = pto.SubTensorType(shape=[M, BASEK], dtype=dtype)
        tile_view_b = pto.SubTensorType(shape=[BASEK, N], dtype=dtype)
        tile_view_out = pto.SubTensorType(shape=[M, N], dtype=dtype)
        tile_view_bias = pto.SubTensorType(shape=[1, N], dtype=dtype)

        tile_buf_aMat = pto.TileBufType(
            shape=[M, BASEK],
            valid_shape=[M, BASEK],
            dtype=dtype,
            memory_space="MAT",
            config=None,
        )
        tile_buf_bMat = pto.TileBufType(
            shape=[BASEK, N],
            valid_shape=[BASEK, N],
            dtype=dtype,
            memory_space="MAT",
            config=None,
        )
        tile_buf_biasData = pto.TileBufType(
            shape=[1, N],
            valid_shape=[1, N],
            dtype=dtype,
            memory_space="MAT",
            config=None,
        )

        tile_buf_aTile = pto.TileBufType(
            shape=[M, BASEK],
            valid_shape=[M, BASEK],
            dtype=dtype,
            memory_space="LEFT",
            config=None,
        )
        tile_buf_bTile = pto.TileBufType(
            shape=[BASEK, N],
            valid_shape=[BASEK, N],
            dtype=dtype,
            memory_space="RIGHT",
            config=None,
        )
        tile_buf_cTile = pto.TileBufType(
            shape=[M, N],
            valid_shape=[M, N],
            dtype=dtype,
            memory_space="ACC",
            config=None,
        )
        tile_buf_biasTile = pto.TileBufType(
            shape=[1, N],
            valid_shape=[1, N],
            dtype=dtype,
            memory_space="BIAS",
            config=None,
        )

        return {
            "ptr_type": ptr_dtype,
            "i1": i1,
            "i32": i32,
            "tv2_a": tv2_a,
            "tv2_b": tv2_b,
            "tv2_out": tv2_out,
            "tv2_bias": tv2_bias,
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

    @jit(meta_data=meta_data, block_dim=1, enable_insert_sync=False)
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

            batch = pto.index_cast(batch_i32)
            cBM = batch * cM

            num_blocks = pto.index_cast(pto.get_block_num())
            batches_per_core = pto.ceil_div(batch, num_blocks)
            bid = pto.index_cast(pto.get_block_idx())
            b_start = bid * batches_per_core
            b_end_unclamped = b_start + batches_per_core
            b_end = pto.min_u(b_end_unclamped, batch)

            tvA = pto.as_tensor(tv2_a, ptr=a_ptr, shape=[cBM, cK], strides=[cK, c1])
            tvB = pto.as_tensor(tv2_b, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
            tvOut = pto.as_tensor(tv2_out, ptr=out_ptr, shape=[cBM, cN], strides=[cN, c1])
            tvBias = pto.as_tensor(tv2_bias, ptr=bias_ptr, shape=[c1, cN], strides=[cN, c1])

            aMatTile = pto.alloc_tile(tile_buf_aMat)
            bMatTile = pto.alloc_tile(tile_buf_bMat)
            biasDataTile = pto.alloc_tile(tile_buf_biasData)
            aTile = pto.alloc_tile(tile_buf_aTile)
            bTile = pto.alloc_tile(tile_buf_bTile)
            cTile = pto.alloc_tile(tile_buf_cTile)
            biasTile = pto.alloc_tile(tile_buf_biasTile)

            for b_idx in pto.for_range(b_start, b_end, c1):
                row_off = b_idx * cM

                for i in pto.for_range(c0, cIter, c1):
                    kOff = i * cBASEK
                    svA = pto.slice_view(
                        tile_view_a,
                        source=tvA,
                        offsets=[row_off, kOff],
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
                    pto.if_else(
                        isBias,
                        lambda: pto.load(svBias, biasDataTile),
                        lambda: None,
                    )

                    pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)

                    pto.mov(aMatTile, aTile)
                    pto.mov(bMatTile, bTile)
                    pto.if_else(
                        isBias,
                        lambda: pto.mov(biasDataTile, biasTile),
                        lambda: None,
                    )

                    pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)

                    is_i0 = pto.eq(i, c0)

                    def _first_iter():
                        pto.if_else(
                            isBias,
                            lambda: pto.matmul_bias(aTile, bTile, biasTile, cTile),
                            lambda: pto.matmul(aTile, bTile, cTile),
                        )

                    pto.if_else(
                        is_i0,
                        _first_iter,
                        lambda: pto.matmul_acc(cTile, aTile, bTile, cTile),
                    )

                    pto.record_wait_pair("MATMUL", "LOAD", event_id=0)

                pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
                svOut = pto.slice_view(
                    tile_view_out,
                    source=tvOut,
                    offsets=[row_off, c0],
                    sizes=[cTileM, cTileN],
                )
                pto.store(cTile, svOut)

    return RunTMATMULSplitK


def test_matmul():
    device = "npu:6"
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    m, k, n = 128, 128, 128
    batch_sizes = [5, 32, 66, 511]
    block_dims = [1, 2, 10, 20]

    torch.manual_seed(0)

    kernel = build_kernel()
    null_ptr = 0
    use_bias = 0

    for bs in batch_sizes:
        a = torch.rand((bs, m, k), device=device, dtype=dtype)
        b = torch.rand((k, n), device=device, dtype=dtype)

        for block_dim in block_dims:
            c = torch.empty((bs, m, n), device=device, dtype=dtype)
            kernel.set_block_dim(block_dim)
            kernel(c, a, b, null_ptr, use_bias, a.shape[0])
            torch.npu.synchronize()

            c_ref = torch.matmul(a, b)
            diff = (c - c_ref).abs().max()
            print(f"config: bs={bs}, block_dim={block_dim}, max diff: {diff}")


if __name__ == "__main__":
    test_matmul()
