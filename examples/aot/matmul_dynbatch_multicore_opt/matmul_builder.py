# adapted from https://github.com/zhangstevenunity/PTOAS/blob/a301aa43b388d9b2e1ba0db8773b3a719e8c445b/test/samples/MatMul/tmatmulk.py

from mlir.ir import (
    Context, Location, InsertionPoint,
    IndexType, IntegerType, F16Type, F32Type, StringAttr
)
from mlir.dialects import func, arith, scf, pto, builtin
from mlir.dialects.pto import ( TLOAD, TMOV_M2L, TMATMUL, TSTORE_ACC, EVENT_ID0)
from mlir.dialects.arith import CmpIPredicate


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build(
    M=128, K=128, N=128,
    validM=128, validK=128, validN=128,
):

    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        module = builtin.ModuleOp()
        module.attributes["pto.device-spec"] = StringAttr.get("Ascend910B1")

        t_out = F32Type.get()
        t_a = F32Type.get()
        t_b = F32Type.get()
        t_bias = F32Type.get()
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)
        ptr_out = pto.PtrType.get(t_out)
        ptr_a = pto.PtrType.get(t_a)
        ptr_b = pto.PtrType.get(t_b)
        ptr_bias = pto.PtrType.get(t_bias)


        # ---- tensor view types ----
        tv2_a = pto.TensorViewType.get(3, t_a)        # [bs, validM, validK]
        tv2_b = pto.TensorViewType.get(2, t_b)        # [validK, validN]
        tv2_out = pto.TensorViewType.get(3, t_out)    # [bs, validM, validN]

        # ---- tile view types ----
        tile_view_a = pto.PartitionTensorViewType.get([M, K], t_a)
        tile_view_b = pto.PartitionTensorViewType.get([K, N], t_b)
        tile_view_out = pto.PartitionTensorViewType.get([M, N], t_out)

        # ---- address spaces ----
        mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT)
        left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT)
        right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT)
        acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC)

        # ---- configs (3rd arg = s_fractal_size) ----
        # MAT tile: used for transfers, typically row_major
        cfg_mat = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null)
        )

        # LEFT tile：TileLeft ... BLayout RowMajor, SLayout RowMajor, fractalAB
        cfg_left = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null)
        )

        # RIGHT tile：TileRight ... BLayout RowMajor, SLayout ColMajor, fractalAB
        cfg_right = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.ColMajor),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null)
        )

        # ACC tile：TileAcc ... BLayout ColMajor, SLayout RowMajor, fractalC
        cfg_acc = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            pto.TileConfig.fractalCSize,
            pto.PadValueAttr.get(pto.PadValue.Null)
        )


        tile_buf_aMat = pto.TileBufType.get([M, K], t_a, mat, [M, K], cfg_mat)
        tile_buf_bMat = pto.TileBufType.get([K, N], t_b, mat, [K, N], cfg_mat)
        tile_buf_aTile = pto.TileBufType.get([M, K], t_a, left, [M, K], cfg_left)
        tile_buf_bTile = pto.TileBufType.get([K, N], t_b, right, [K, N], cfg_right)
        tile_buf_cTile = pto.TileBufType.get([M, N], t_out, acc, [M,N], cfg_acc)

        # ---- function ----
        # (out, A, B, bias, isBias, batchSize)
        fn_ty = func.FunctionType.get([ptr_out, ptr_a, ptr_b, ptr_bias, i1, i32], [])
        with InsertionPoint(module.body):
            fn = func.FuncOp("RunTMATMULSplitK", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            """
            For now we assume A: [bs, M, K], B: [K, N], C: [bs, M, N] and we do batched matmul (B is broadcasted).
            MKN is known at compile-time while bs is not.


            Since every multiplcation uses the same B, just load it once GM -> L1 -> L0B.
            """
            out_ptr, a_ptr, b_ptr, bias_ptr, isBias, batch_i32 = entry.arguments

            cube_section = pto.SectionCubeOp()
            cube_block = cube_section.body.blocks.append()

            with InsertionPoint(cube_block):
                c0 = _idx_const(0)
                c1 = _idx_const(1)
                cM = _idx_const(validM)
                cK = _idx_const(validK)
                cN = _idx_const(validN)
                cKM = _idx_const(validK*validM)
                cMN = _idx_const(validM*validN)
                cTileM = _idx_const(M)
                cTileN = _idx_const(N)

                batch = arith.IndexCastOp(IndexType.get(), batch_i32).result
                # Total logical rows for A/C: [batch, M, *] -> [batch*M, *]
                cBM = arith.MulIOp(batch, cM).result

                # Distribute batches over cores. Each core gets B//C, and B%C cores gets +1
                num_blocks = arith.IndexCastOp(IndexType.get(), pto.GetBlockNumOp().result).result
                bid        = arith.IndexCastOp(IndexType.get(), pto.GetBlockIdxOp().result).result

                base = arith.DivSIOp(batch, num_blocks).result
                rem  = arith.RemSIOp(batch, num_blocks).result
                lt_rem = arith.CmpIOp(arith.CmpIPredicate.slt, bid, rem).result
                min_bid_rem = arith.MinUIOp(bid, rem).result
                b_start = arith.AddIOp(arith.MulIOp(bid, base).result, min_bid_rem).result
                length = arith.AddIOp(base, arith.SelectOp(lt_rem, c1, c0).result).result
                b_end = arith.MinUIOp(arith.AddIOp(b_start, length).result, batch).result

                # ---- make_tensor_view over full [batch*M, *] range ----
                # A: [batch, M, validK], stride [K*M, K, 1]
                # B: [validK, validN], stride [validN, 1] (shared across batches)
                # OUT: [batch, M, validN], stride [M*N, N, 1]
                tvA = pto.MakeTensorViewOp(tv2_a, a_ptr, [batch, cM, cK], [cKM, cK, c1]).result
                tvB = pto.MakeTensorViewOp(tv2_b, b_ptr, [cK, cN], [cN, c1]).result
                tvOut = pto.MakeTensorViewOp(tv2_out, out_ptr, [batch, cM, cN], [cMN, cN, c1]).result

                # ---- alloc tiles ----
                aMatTile = pto.AllocTileOp(tile_buf_aMat).result
                bMatTile = pto.AllocTileOp(tile_buf_bMat).result
                aTile = pto.AllocTileOp(tile_buf_aTile).result
                bTile = pto.AllocTileOp(tile_buf_bTile).result
                cTile = pto.AllocTileOp(tile_buf_cTile).result

                svB = pto.PartitionViewOp(tile_view_b, tvB, offsets=[c0, c0], sizes=[cK, cTileN]).result
                # Load GM into tile
                pto.TLoadOp(None, svB, bMatTile)
                pto.record_event(TLOAD, TMOV_M2L, EVENT_ID0)
                pto.wait_event  (TLOAD, TMOV_M2L, EVENT_ID0)
                # move from L1 to L0
                pto.TMovOp(None, bMatTile, bTile)

                # ---- outer loop over batches assigned to this core ----
                batch_loop = scf.ForOp(b_start, b_end, c1)
                with InsertionPoint(batch_loop.body):
                    b_idx = batch_loop.induction_variable

                    # subviews for this batch row range
                    svA = pto.PartitionViewOp(tile_view_a, tvA, offsets=[b_idx, c0, c0], sizes=[c1, cTileM, cK]).result
                    svOut = pto.PartitionViewOp(tile_view_out, tvOut, offsets=[b_idx, c0, c0], sizes=[c1, cTileM, cTileN]).result

                    # Note: TLOAD valid dims typically correspond to the destination tile's valid region (a/b/bias)
                    pto.TLoadOp(None, svA, aMatTile)

                    # ---- sync: MTE2 -> MTE1 ----
                    pto.record_event(TLOAD, TMOV_M2L, EVENT_ID0)
                    pto.wait_event  (TLOAD, TMOV_M2L, EVENT_ID0)

                    # TMOV also uses the corresponding tile valid dims (a/b/bias)
                    pto.TMovOp(None, aMatTile, aTile)

                    # ---- sync: MTE1 -> M ----
                    pto.record_event(TMOV_M2L, TMATMUL, EVENT_ID0)
                    pto.wait_event  (TMOV_M2L, TMATMUL, EVENT_ID0)
                    pto.TMatmulOp(None, aTile, bTile, cTile)
                    pto.record_event(TMATMUL, TLOAD, EVENT_ID0)
                    pto.wait_event  (TMATMUL, TLOAD, EVENT_ID0)

                    # ---- after split-K loop for this batch ----
                    pto.record_event(TMATMUL, TSTORE_ACC, EVENT_ID0)
                    pto.wait_event  (TMATMUL, TSTORE_ACC, EVENT_ID0)

                    # Write back OUT using the valid dims of C
                    pto.TStoreOp(None, cTile, svOut)

                    scf.YieldOp([])

            func.ReturnOp([])
        module.operation.verify()
        return module


if __name__ == "__main__":
    m = build()
    print(m)