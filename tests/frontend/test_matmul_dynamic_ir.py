import importlib.util
from pathlib import Path

from mlir.dialects import arith, func, pto, scf
from mlir.dialects.arith import CmpIPredicate
from mlir.dialects.pto import EVENT_ID0, TLOAD, TMATMUL, TMOV_M2L, TSTORE_ACC
from mlir.ir import Context, F32Type, IndexType, InsertionPoint, IntegerType, Location, Module


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def _load_pythonic_builder():
    root = Path(__file__).resolve().parents[2]
    builder_path = root / "examples" / "aot" / "matmul_dynbatch_multicore" / "matmul_builder.py"
    spec = importlib.util.spec_from_file_location("matmul_builder_module", builder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_verbose(
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

    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)
        module = Module.create()

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

        tv2_a = pto.TensorViewType.get(2, t_a)
        tv2_b = pto.TensorViewType.get(2, t_b)
        tv2_out = pto.TensorViewType.get(2, t_out)
        tv2_bias = pto.TensorViewType.get(2, t_bias)

        tile_view_a = pto.PartitionTensorViewType.get([M, BASEK], t_a)
        tile_view_b = pto.PartitionTensorViewType.get([BASEK, N], t_b)
        tile_view_out = pto.PartitionTensorViewType.get([M, N], t_out)
        tile_view_bias = pto.PartitionTensorViewType.get([1, N], t_bias)

        mat = pto.AddressSpaceAttr.get(pto.AddressSpace.MAT)
        left = pto.AddressSpaceAttr.get(pto.AddressSpace.LEFT)
        right = pto.AddressSpaceAttr.get(pto.AddressSpace.RIGHT)
        acc = pto.AddressSpaceAttr.get(pto.AddressSpace.ACC)
        bias = pto.AddressSpaceAttr.get(pto.AddressSpace.BIAS)

        cfg_mat = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )
        cfg_mat_bias = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.NoneBox),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )
        cfg_left = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )
        cfg_right = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.ColMajor),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )
        cfg_acc = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.ColMajor),
            pto.SLayoutAttr.get(pto.SLayout.RowMajor),
            pto.TileConfig.fractalCSize,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )
        cfg_bias = pto.TileBufConfigAttr.get(
            pto.BLayoutAttr.get(pto.BLayout.RowMajor),
            pto.SLayoutAttr.get(pto.SLayout.NoneBox),
            pto.TileConfig.fractalABSize,
            pto.PadValueAttr.get(pto.PadValue.Null),
        )

        tile_buf_aMat = pto.TileBufType.get([M, BASEK], t_a, mat, [M, BASEK], cfg_mat)
        tile_buf_bMat = pto.TileBufType.get([BASEK, N], t_b, mat, [BASEK, N], cfg_mat)
        tile_buf_biasData = pto.TileBufType.get([1, N], t_bias, mat, [1, N], cfg_mat_bias)
        tile_buf_aTile = pto.TileBufType.get([M, BASEK], t_a, left, [M, BASEK], cfg_left)
        tile_buf_bTile = pto.TileBufType.get([BASEK, N], t_b, right, [BASEK, N], cfg_right)
        tile_buf_cTile = pto.TileBufType.get([M, N], t_out, acc, [M, N], cfg_acc)
        tile_buf_biasTile = pto.TileBufType.get([1, N], t_bias, bias, [1, N], cfg_bias)

        fn_ty = func.FunctionType.get([ptr_out, ptr_a, ptr_b, ptr_bias, i1, i32], [])
        with InsertionPoint(module.body):
            fn = func.FuncOp("RunTMATMULSplitK", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            out_ptr, a_ptr, b_ptr, bias_ptr, isBias, batch_i32 = entry.arguments

            cube_section = pto.SectionCubeOp()
            cube_block = cube_section.body.blocks.append()

            with InsertionPoint(cube_block):
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
                cBM = arith.MulIOp(batch, cM).result

                num_blocks = arith.IndexCastOp(IndexType.get(), pto.GetBlockNumOp().result).result
                batches_per_core = arith.CeilDivSIOp(batch, num_blocks).result
                bid = arith.IndexCastOp(IndexType.get(), pto.GetBlockIdxOp().result).result
                b_start = arith.MulIOp(bid, batches_per_core).result
                b_end_unclamped = arith.AddIOp(b_start, batches_per_core).result
                b_end = arith.MinUIOp(b_end_unclamped, batch).result

                tvA = pto.MakeTensorViewOp(tv2_a, a_ptr, [cBM, cK], [cK, c1]).result
                tvB = pto.MakeTensorViewOp(tv2_b, b_ptr, [cK, cN], [cN, c1]).result
                tvOut = pto.MakeTensorViewOp(tv2_out, out_ptr, [cBM, cN], [cN, c1]).result
                tvBias = pto.MakeTensorViewOp(tv2_bias, bias_ptr, [c1, cN], [cN, c1]).result

                aMatTile = pto.AllocTileOp(tile_buf_aMat).result
                bMatTile = pto.AllocTileOp(tile_buf_bMat).result
                biasDataTile = pto.AllocTileOp(tile_buf_biasData).result
                aTile = pto.AllocTileOp(tile_buf_aTile).result
                bTile = pto.AllocTileOp(tile_buf_bTile).result
                cTile = pto.AllocTileOp(tile_buf_cTile).result
                biasTile = pto.AllocTileOp(tile_buf_biasTile).result

                batch_loop = scf.ForOp(b_start, b_end, c1)
                with InsertionPoint(batch_loop.body):
                    b_idx = batch_loop.induction_variable
                    row_off = arith.MulIOp(b_idx, cM).result

                    for i in scf.for_(c0, cIter, c1):
                        kOff = arith.MulIOp(i, cBASEK).result
                        svA = pto.PartitionViewOp(
                            tile_view_a, tvA, offsets=[row_off, kOff], sizes=[cTileM, cBASEK]
                        ).result
                        svB = pto.PartitionViewOp(
                            tile_view_b, tvB, offsets=[kOff, c0], sizes=[cBASEK, cTileN]
                        ).result
                        svBias = pto.PartitionViewOp(
                            tile_view_bias, tvBias, offsets=[c0, c0], sizes=[c1, cTileN]
                        ).result

                        pto.TLoadOp(None, svA, aMatTile)
                        pto.TLoadOp(None, svB, bMatTile)

                        if_load_bias = scf.IfOp(isBias, [], hasElse=True)
                        with InsertionPoint(if_load_bias.then_block):
                            pto.TLoadOp(None, svBias, biasDataTile)
                            scf.YieldOp([])
                        with InsertionPoint(if_load_bias.else_block):
                            scf.YieldOp([])

                        pto.record_event(TLOAD, TMOV_M2L, EVENT_ID0)
                        pto.wait_event(TLOAD, TMOV_M2L, EVENT_ID0)

                        pto.TMovOp(None, aMatTile, aTile)
                        pto.TMovOp(None, bMatTile, bTile)

                        if_mov_bias = scf.IfOp(isBias, [], hasElse=True)
                        with InsertionPoint(if_mov_bias.then_block):
                            pto.TMovOp(None, biasDataTile, biasTile)
                            scf.YieldOp([])
                        with InsertionPoint(if_mov_bias.else_block):
                            scf.YieldOp([])

                        pto.record_event(TMOV_M2L, TMATMUL, EVENT_ID0)
                        pto.wait_event(TMOV_M2L, TMATMUL, EVENT_ID0)

                        is_i0 = arith.CmpIOp(CmpIPredicate.eq, i, c0).result
                        if_i0 = scf.IfOp(is_i0, [], hasElse=True)
                        with InsertionPoint(if_i0.then_block):
                            if_bias0 = scf.IfOp(isBias, [], hasElse=True)
                            with InsertionPoint(if_bias0.then_block):
                                pto.TMatmulBiasOp(None, aTile, bTile, biasTile, cTile)
                                scf.YieldOp([])
                            with InsertionPoint(if_bias0.else_block):
                                pto.TMatmulOp(None, aTile, bTile, cTile)
                                scf.YieldOp([])
                            scf.YieldOp([])
                        with InsertionPoint(if_i0.else_block):
                            pto.TMatmulAccOp(None, cTile, aTile, bTile, cTile)
                            scf.YieldOp([])

                        pto.record_event(TMATMUL, TLOAD, EVENT_ID0)
                        pto.wait_event(TMATMUL, TLOAD, EVENT_ID0)
                        scf.YieldOp([])

                    pto.record_event(TMATMUL, TSTORE_ACC, EVENT_ID0)
                    pto.wait_event(TMATMUL, TSTORE_ACC, EVENT_ID0)
                    svOut = pto.PartitionViewOp(
                        tile_view_out, tvOut, offsets=[row_off, c0], sizes=[cTileM, cTileN]
                    ).result
                    pto.TStoreOp(None, cTile, svOut)
                    scf.YieldOp([])

            func.ReturnOp([])

        module.operation.verify()
        return module


def test_matmul_structural_ir_equality():
    pythonic_module = _load_pythonic_builder().build()
    verbose_module = build_verbose()
    assert str(pythonic_module) == str(verbose_module)
