# adapted from https://github.com/zhangstevenunity/PTOAS/blob/a301aa43b388d9b2e1ba0db8773b3a719e8c445b/test/samples/MatMul/tmatmulk.py

from mlir.ir import (
    Context, Location, InsertionPoint,
    IndexType, IntegerType, F16Type, F32Type, StringAttr
)
from mlir.dialects import func, arith, scf, pto, builtin
from mlir.dialects.pto import (
    TLOAD, TMOV_M2L, TMATMUL, TSTORE_ACC,
    EVENT_ID0
)
from mlir.dialects.arith import CmpIPredicate


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build(
    M=128, K=128, N=128,
    validM=128, validK=128, validN=128,
    BASEK=32,
    s_fractal_ab=512,
    s_fractal_c=1024,
):
    assert K % BASEK == 0
    iters = K // BASEK

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
        tv2_a = pto.TensorViewType.get(2, t_a)        # [validM, validK]
        tv2_b = pto.TensorViewType.get(2, t_b)        # [validK, validN]
        tv2_out = pto.TensorViewType.get(2, t_out)    # [validM, validN]
        tv2_bias = pto.TensorViewType.get(2, t_bias)  # [1, validN]

        # ---- tile view types ----
        tile_view_a = pto.PartitionTensorViewType.get([M, BASEK], t_a)
        tile_view_b = pto.PartitionTensorViewType.get([BASEK, N], t_b)
        tile_view_out = pto.PartitionTensorViewType.get([M, N], t_out)
        tile_view_bias = pto.PartitionTensorViewType.get([1, N], t_bias)

        # ---- address spaces ----
        mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT)
        left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT)
        right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT)
        acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC)
        bias = pto.AddressSpaceAttr.get(pto.AddressSpace.BIAS)

        # ---- configs (3rd arg = s_fractal_size) ----
        # Note: these layout/pad settings are reasonable defaults; adjust to match your C++ Tile definitions if needed.
        # MAT tile: used for transfers, typically row_major
        cfg_mat = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_ab,  # 这里也可以单独给 MAT 一个 size
            pto.PadValueAttr.get(pto.PadValue.Null)
        )

        cfg_mat_bias = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.NoneBox),
            s_fractal_ab,  # 这里也可以单独给 MAT 一个 size
            pto.PadValueAttr.get(pto.PadValue.Null)
        )


        # LEFT tile：TileLeft ... BLayout RowMajor, SLayout RowMajor, fractalAB
        cfg_left = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_ab,
            pto.PadValueAttr.get(pto.PadValue.Null)
        )

        # RIGHT tile：TileRight ... BLayout RowMajor, SLayout ColMajor, fractalAB
        cfg_right = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.ColMajor),
            s_fractal_ab,
            pto.PadValueAttr.get(pto.PadValue.Null)
        )

        # ACC tile：TileAcc ... BLayout ColMajor, SLayout RowMajor, fractalC
        cfg_acc = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            s_fractal_c,
            pto.PadValueAttr.get(pto.PadValue.Null)
        )

        # BIAS tile: usually does not require fractalization (we use a default fractal size here; a dedicated size is also possible)
        cfg_bias = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.NoneBox),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null)
        )

        # ---- tile buf types (each has its own cfg) ----
        tile_buf_aMat = pto.TileBufType.get([M, BASEK], t_a, mat, [M, BASEK], cfg_mat)
        tile_buf_bMat = pto.TileBufType.get([BASEK, N], t_b, mat, [BASEK, N], cfg_mat)
        tile_buf_biasData = pto.TileBufType.get([1, N], t_bias, mat, [1, N], cfg_mat_bias)

        tile_buf_aTile = pto.TileBufType.get([M, BASEK], t_a, left, [M, BASEK], cfg_left)
        tile_buf_bTile = pto.TileBufType.get([BASEK, N], t_b, right, [BASEK, N], cfg_right)
        tile_buf_cTile = pto.TileBufType.get([M, N], t_out, acc, [M,N], cfg_acc)
        tile_buf_biasTile = pto.TileBufType.get([1, N], t_bias, bias, [1, N], cfg_bias)

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

            The reduction over the k dim is done in chunks of size BASEK. (So for A tiles are [M, BASEK] and for B: [BASEK, N])

            Each core c=0,1,...,C-1 get allocated ceil(bs, C) batches (this is bad load-balancing for e.g. bs=9 C=8 since some cores does 0)


            every multiplcation will use the same B, so make sure we keep it in fast mem
            """
            out_ptr, a_ptr, b_ptr, bias_ptr, isBias, batch_i32 = entry.arguments

            cube_section = pto.SectionCubeOp()
            cube_block = cube_section.body.blocks.append()

            with InsertionPoint(cube_block):
                # ---- constants ----
                c0 = _idx_const(0)
                c1 = _idx_const(1)

                cM = _idx_const(validM)
                cK = _idx_const(validK)
                cN = _idx_const(validN)

                cBASEK = _idx_const(BASEK)
                cIter = _idx_const(iters)

                cTileM = _idx_const(M)
                cTileN = _idx_const(N)

                batch = arith.IndexCastOp(IndexType.get(), batch_i32).result
                # Total logical rows for A/C: [batch, M, *] -> [batch*M, *]
                cBM = arith.MulIOp(batch, cM).result

                # ---- batch partitioning across blocks ----
                # Assign ceil(batch / numBlocks) batches per core.
                num_blocks = arith.IndexCastOp(IndexType.get(), pto.GetBlockNumOp().result).result
                batches_per_core = arith.CeilDivSIOp(batch, num_blocks).result
                bid = arith.IndexCastOp(IndexType.get(), pto.GetBlockIdxOp().result).result

                b_start = arith.MulIOp(bid, batches_per_core).result
                b_end_unclamped = arith.AddIOp(b_start, batches_per_core).result
                b_end = arith.MinUIOp(b_end_unclamped, batch).result


                # ---- make_tensor_view over full [batch*M, *] range ----
                # A: [batch*M, validK], stride [validK, 1]
                tvA = pto.MakeTensorViewOp(tv2_a, a_ptr, [cBM, cK], [cK, c1]).result
                # B: [validK, validN], stride [validN, 1] (shared across batches)
                tvB = pto.MakeTensorViewOp(tv2_b, b_ptr, [cK, cN], [cN, c1]).result
                # OUT: [batch*M, validN], stride [validN, 1]
                tvOut = pto.MakeTensorViewOp(tv2_out, out_ptr, [cBM, cN], [cN, c1]).result
                # BIAS: [1, validN], stride [validN, 1]
                tvBias = pto.MakeTensorViewOp(tv2_bias, bias_ptr, [c1, cN], [cN, c1]).result

                # ---- alloc tiles ----
                aMatTile = pto.AllocTileOp(tile_buf_aMat).result
                bMatTile = pto.AllocTileOp(tile_buf_bMat).result
                biasDataTile = pto.AllocTileOp(tile_buf_biasData).result

                aTile = pto.AllocTileOp(tile_buf_aTile).result
                bTile = pto.AllocTileOp(tile_buf_bTile).result
                cTile = pto.AllocTileOp(tile_buf_cTile).result
                biasTile = pto.AllocTileOp(tile_buf_biasTile).result

                # ---- valid dims (passed into ops; alloc has no valid operands) ----
                # Align with the C++ TileLeft/Right/Acc/Bias RowValid_/ColValid_ semantics

                # ---- outer loop over batches assigned to this core ----
                batch_loop = scf.ForOp(b_start, b_end, c1)
                with InsertionPoint(batch_loop.body):
                    b_idx = batch_loop.induction_variable

                    # Row offset for this batch within the [batch*M, *] views.
                    row_off = arith.MulIOp(b_idx, cM).result

                    # ---- loop for split-K ----
                    for i in scf.for_(c0, cIter, c1):
                        # kOff = i * BASEK
                        kOff = arith.MulIOp(i, cBASEK).result

                        # subviews for this split-K and batch row range
                        svA = pto.PartitionViewOp(tile_view_a, tvA, offsets=[row_off, kOff], sizes=[cTileM, cBASEK]).result
                        svB = pto.PartitionViewOp(tile_view_b, tvB, offsets=[kOff, c0], sizes=[cBASEK, cTileN]).result
                        svBias = pto.PartitionViewOp(tile_view_bias, tvBias, offsets=[c0, c0], sizes=[c1, cTileN]).result

                        # ---- TLOAD ----
                        # Note: TLOAD valid dims typically correspond to the destination tile's valid region (a/b/bias)
                        pto.TLoadOp(None, svA, aMatTile)
                        pto.TLoadOp(None, svB, bMatTile)

                        if_load_bias = scf.IfOp(isBias, [], hasElse=True)
                        with InsertionPoint(if_load_bias.then_block):
                            pto.TLoadOp(None, svBias, biasDataTile)
                            scf.YieldOp([])
                        with InsertionPoint(if_load_bias.else_block):
                            scf.YieldOp([])

                        # ---- sync: MTE2 -> MTE1 ----
                        pto.record_event(TLOAD, TMOV_M2L, EVENT_ID0)
                        pto.wait_event  (TLOAD, TMOV_M2L, EVENT_ID0)

                        # ---- TMOV ----
                        # TMOV also uses the corresponding tile valid dims (a/b/bias)
                        pto.TMovOp(None, aMatTile, aTile)
                        pto.TMovOp(None, bMatTile, bTile)

                        if_mov_bias = scf.IfOp(isBias, [], hasElse=True)
                        with InsertionPoint(if_mov_bias.then_block):
                            pto.TMovOp(None, biasDataTile, biasTile)
                            scf.YieldOp([])
                        with InsertionPoint(if_mov_bias.else_block):
                            scf.YieldOp([])

                        # ---- sync: MTE1 -> M ----
                        pto.record_event(TMOV_M2L, TMATMUL, EVENT_ID0)
                        pto.wait_event  (TMOV_M2L, TMATMUL, EVENT_ID0)

                        # ---- i == 0 ? (bias? TMATMUL_BIAS : TMATMUL) : TMATMUL_ACC ----
                        is_i0 = arith.CmpIOp(CmpIPredicate.eq, i, c0).result
                        if_i0 = scf.IfOp(is_i0, [], hasElse=True)

                        # then: i == 0
                        with InsertionPoint(if_i0.then_block):
                            if_bias0 = scf.IfOp(isBias, [], hasElse=True)
                            with InsertionPoint(if_bias0.then_block):
                                # Clear accumulator tile and apply bias
                                # Convention: valid_dims_c describes the valid region of C
                                pto.TMatmulBiasOp(None,aTile, bTile, biasTile, cTile)
                                scf.YieldOp([])
                            with InsertionPoint(if_bias0.else_block):
                                # Clear accumulator tile without bias
                                pto.TMatmulOp(None, aTile, bTile, cTile)
                                scf.YieldOp([])
                            scf.YieldOp([])

                        # else: i != 0
                        with InsertionPoint(if_i0.else_block):
                            # Do not clear accumulator tile; accumulate into existing C
                            pto.TMatmulAccOp(None, cTile, aTile, bTile, cTile)
                            scf.YieldOp([])

                        # ---- sync: M -> MTE2 ----
                        pto.record_event(TMATMUL, TLOAD, EVENT_ID0)
                        pto.wait_event  (TMATMUL, TLOAD, EVENT_ID0)

                        scf.YieldOp([])

                    # ---- after split-K loop for this batch ----
                    pto.record_event(TMATMUL, TSTORE_ACC, EVENT_ID0)
                    pto.wait_event  (TMATMUL, TSTORE_ACC, EVENT_ID0)

                    # ---- TSTORE ----
                    # Write back OUT using the valid dims of C
                    svOut = pto.PartitionViewOp(tile_view_out, tvOut, offsets=[row_off, c0], sizes=[cTileM, cTileN]).result
                    pto.TStoreOp(None, cTile, svOut)

                    scf.YieldOp([])

            func.ReturnOp([])
        module.operation.verify()
        return module


if __name__ == "__main__":
    m = build()
    print(m)