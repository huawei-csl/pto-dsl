from mlir.ir import Context, Location, Module, InsertionPoint, IntegerType
from mlir.ir import F32Type, IndexType
from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const


def meta_data():
    dtype = pto.float32
    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=2, dtype=dtype)
    subtensor_type = pto.SubTensorType(shape=[32, 32], dtype=dtype)
    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[32, 32], valid_shape=[-1, -1], dtype=dtype, memory_space="VEC", config=tile_cfg
    )
    return {
        "ptr_type": ptr_type,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_type": subtensor_type,
        "tile_type": tile_type,
    }


def vec_add_kernel_2d_dynamic(
    arg0: "ptr_type",
    arg1: "ptr_type",
    arg2: "ptr_type",
    arg_vrow_i32: "index_dtype",
    arg_vcol_i32: "index_dtype",
) -> None:
    c0 = const(0)
    c1 = const(1)
    c32 = const(32)
    c1280 = const(1280)

    cid = pto.get_block_idx()
    sub_bid = pto.get_subblock_idx()
    sub_bnum = pto.get_subblock_num()
    cidmul = cid * sub_bnum
    vid = cidmul + sub_bid

    v_row_idx = pto.index_cast(arg_vrow_i32)
    v_col_idx = pto.index_cast(arg_vcol_i32)

    tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[c1280, c32], strides=[c32, c1])
    tv1 = pto.as_tensor(tensor_type, ptr=arg1, shape=[c1280, c32], strides=[c32, c1])
    tv2 = pto.as_tensor(tensor_type, ptr=arg2, shape=[c1280, c32], strides=[c32, c1])

    vid_idx = pto.index_cast(vid)
    offset_row = vid_idx * c32
    sv0 = pto.slice_view(subtensor_type, source=tv0, offsets=[offset_row, c0], sizes=[c32, c32])
    sv1 = pto.slice_view(subtensor_type, source=tv1, offsets=[offset_row, c0], sizes=[c32, c32])
    sv2 = pto.slice_view(subtensor_type, source=tv2, offsets=[offset_row, c0], sizes=[c32, c32])

    with pto.vector_section():
        tb0 = pto.alloc_tile(tile_type, valid_row=v_row_idx, valid_col=v_col_idx)
        tb1 = pto.alloc_tile(tile_type, valid_row=v_row_idx, valid_col=v_col_idx)
        tb2 = pto.alloc_tile(tile_type, valid_row=v_row_idx, valid_col=v_col_idx)

        pto.load(sv0, tb0)
        pto.load(sv1, tb1)
        pto.add(tb0, tb1, tb2)
        pto.store(tb2, sv2)


def build():
    from mlir.dialects import func, arith, pto

    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        m = Module.create()

        f32 = F32Type.get()
        i32 = IntegerType.get_signless(32)
        ptr_f32 = pto.PtrType.get(f32)

        tv2_f32 = pto.TensorViewType.get(2, f32)
        tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32)
        vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC)
        bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor)
        sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox)
        pd = pto.PadValueAttr.get(pto.PadValue.Null)

        cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd)

        tile_buf_dynamic = pto.TileBufType.get([32, 32], f32, vec, [-1, -1], cfg)
        fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32, i32, i32], [])

        with InsertionPoint(m.body):
            fn = func.FuncOp("vec_add_kernel_2d_dynamic", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            c0 = arith.ConstantOp(IndexType.get(), 0).result
            c1 = arith.ConstantOp(IndexType.get(), 1).result
            c32 = arith.ConstantOp(IndexType.get(), 32).result
            c1280 = arith.ConstantOp(IndexType.get(), 1280).result

            arg0, arg1, arg2, arg_vrow_i32, arg_vcol_i32 = entry.arguments

            cid = pto.GetBlockIdxOp().result
            sub_bid = pto.GetSubBlockIdxOp().result
            sub_bnum = pto.GetSubBlockNumOp().result
            cidmul = arith.MulIOp(cid, sub_bnum).result
            vid = arith.AddIOp(cidmul, sub_bid).result

            v_row_idx = arith.IndexCastOp(IndexType.get(), arg_vrow_i32).result
            v_col_idx = arith.IndexCastOp(IndexType.get(), arg_vcol_i32).result

            tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c1280, c32], [c32, c1]).result
            tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c1280, c32], [c32, c1]).result
            tv2 = pto.MakeTensorViewOp(tv2_f32, arg2, [c1280, c32], [c32, c1]).result

            vid_idx = arith.IndexCastOp(IndexType.get(), vid).result
            offset_row = arith.MulIOp(vid_idx, c32).result
            sv0 = pto.PartitionViewOp(
                tile_view_32, tv0, offsets=[offset_row, c0], sizes=[c32, c32]
            ).result
            sv1 = pto.PartitionViewOp(
                tile_view_32, tv1, offsets=[offset_row, c0], sizes=[c32, c32]
            ).result
            sv2 = pto.PartitionViewOp(
                tile_view_32, tv2, offsets=[offset_row, c0], sizes=[c32, c32]
            ).result

            vec_section = pto.SectionVectorOp()
            vec_block = vec_section.body.blocks.append()
            with InsertionPoint(vec_block):
                tb0 = pto.AllocTileOp(tile_buf_dynamic, valid_row=v_row_idx, valid_col=v_col_idx).result
                tb1 = pto.AllocTileOp(tile_buf_dynamic, valid_row=v_row_idx, valid_col=v_col_idx).result
                tb2 = pto.AllocTileOp(tile_buf_dynamic, valid_row=v_row_idx, valid_col=v_col_idx).result

                pto.TLoadOp(None, sv0, tb0)
                pto.TLoadOp(None, sv1, tb1)
                pto.TAddOp(tb0, tb1, tb2)
                pto.TStoreOp(None, tb2, sv2)

            func.ReturnOp([])

        m.operation.verify()
        return m


def test_structural_ir_equality():
    # NOTE: function name also need to match
    dsl_module = to_ir_module(meta_data=meta_data)(vec_add_kernel_2d_dynamic)
    ref_module = build()
    assert str(dsl_module) == str(ref_module)
