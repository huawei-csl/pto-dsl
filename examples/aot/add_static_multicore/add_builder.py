from mlir.ir import Context, Location, Module, InsertionPoint, IntegerType
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType

def build():
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
            c1280 = arith.ConstantOp(IndexType.get(), 1280).result  # 32 per-core x 40 cores

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
            offset_row = arith.MulIOp(vid_idx, c32).result  # every core loads 32 rows of data
            sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[offset_row, c0], sizes=[c32, c32]).result
            sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[offset_row, c0], sizes=[c32, c32]).result
            sv2 = pto.PartitionViewOp(tile_view_32, tv2, offsets=[offset_row, c0], sizes=[c32, c32]).result

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

if __name__ == "__main__":
    print(build())
