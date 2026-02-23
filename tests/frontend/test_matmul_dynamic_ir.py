from mlir.dialects import arith, func, pto as _pto, scf
from mlir.dialects.arith import CmpIPredicate
from mlir.dialects.pto import EVENT_ID0, TLOAD, TMATMUL, TMOV_M2L, TSTORE_ACC
from mlir.ir import Context, F32Type, IndexType, InsertionPoint, IntegerType, Location, Module
from ptodsl import to_ir_module
import ptodsl.language as pto


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build_pythonic(
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
        tile_buf_biasTile = pto.TileBufType(shape=[1, N], dtype=dtype, memory_space="BIAS")

        return {
            "ptr_type": ptr_dtype,
            "i1": i1,
            "i32": i32,
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

            tvA = pto.as_tensor(tensor_type, ptr=a_ptr, shape=[cBM, cK], strides=[cK, c1])
            tvB = pto.as_tensor(tensor_type, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
            tvOut = pto.as_tensor(tensor_type, ptr=out_ptr, shape=[cBM, cN], strides=[cN, c1])
            tvBias = pto.as_tensor(tensor_type, ptr=bias_ptr, shape=[c1, cN], strides=[cN, c1])

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
                    pto.if_else(isBias, lambda: pto.load(svBias, biasDataTile), lambda: None)

                    pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)

                    pto.mov(aMatTile, aTile)
                    pto.mov(bMatTile, bTile)
                    pto.if_else(isBias, lambda: pto.mov(biasDataTile, biasTile), lambda: None)

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

    pto = _pto
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)
        module = Module.create()

        dtype = F32Type.get()
        ptr_dtype = pto.PtrType.get(dtype)
        i1 = IntegerType.get_signless(1)
        i32 = IntegerType.get_signless(32)

        tensor_type = pto.TensorViewType.get(2, dtype)

        tile_view_a = pto.PartitionTensorViewType.get([M, BASEK], dtype)
        tile_view_b = pto.PartitionTensorViewType.get([BASEK, N], dtype)
        tile_view_out = pto.PartitionTensorViewType.get([M, N], dtype)
        tile_view_bias = pto.PartitionTensorViewType.get([1, N], dtype)

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

        tile_buf_aMat = pto.TileBufType.get([M, BASEK], dtype, mat, [M, BASEK], cfg_mat)
        tile_buf_bMat = pto.TileBufType.get([BASEK, N], dtype, mat, [BASEK, N], cfg_mat)
        tile_buf_biasData = pto.TileBufType.get([1, N], dtype, mat, [1, N], cfg_mat_bias)
        tile_buf_aTile = pto.TileBufType.get([M, BASEK], dtype, left, [M, BASEK], cfg_left)
        tile_buf_bTile = pto.TileBufType.get([BASEK, N], dtype, right, [BASEK, N], cfg_right)
        tile_buf_cTile = pto.TileBufType.get([M, N], dtype, acc, [M, N], cfg_acc)
        tile_buf_biasTile = pto.TileBufType.get([1, N], dtype, bias, [1, N], cfg_bias)

        fn_ty = func.FunctionType.get([ptr_dtype, ptr_dtype, ptr_dtype, ptr_dtype, i1, i32], [])
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

                tvA = pto.MakeTensorViewOp(tensor_type, a_ptr, [cBM, cK], [cK, c1]).result
                tvB = pto.MakeTensorViewOp(tensor_type, b_ptr, [cK, cN], [cN, c1]).result
                tvOut = pto.MakeTensorViewOp(tensor_type, out_ptr, [cBM, cN], [cN, c1]).result
                tvBias = pto.MakeTensorViewOp(tensor_type, bias_ptr, [c1, cN], [cN, c1]).result

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
    pythonic_module = build_pythonic()
    verbose_module = build_verbose()
    assert str(pythonic_module) == str(verbose_module)
