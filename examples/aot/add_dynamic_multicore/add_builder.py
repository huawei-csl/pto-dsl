from mlir.ir import Context, Location, Module, InsertionPoint, IntegerType
from mlir.ir import F32Type, IndexType
from mlir.dialects import func, arith, scf, pto


def build():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        m = Module.create()

        f32 = F32Type.get()
        u32 = IntegerType.get_signless(32)
        ptr_f32 = pto.PtrType.get(f32)

        tv1_f32 = pto.TensorViewType.get(1, f32)

        tile_length = 1024
        tile_view_32 = pto.PartitionTensorViewType.get([1, tile_length], f32)
        vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC)
        bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor)
        sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox)
        pd = pto.PadValueAttr.get(pto.PadValue.Null)

        cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd)

        tile_buf = pto.TileBufType.get([1, tile_length], f32, vec, [1, tile_length], cfg)
        fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32, u32], [])

        with InsertionPoint(m.body):
            fn = func.FuncOp("vec_add_1d_dynamic", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            c0 = arith.ConstantOp(IndexType.get(), 0).result
            c1 = arith.ConstantOp(IndexType.get(), 1).result
            c1024 = arith.ConstantOp(IndexType.get(), tile_length).result

            arg0, arg1, arg2, argN = entry.arguments

            cid = pto.GetBlockIdxOp().result
            sub_bid = pto.GetSubBlockIdxOp().result
            sub_bnum = pto.GetSubBlockNumOp().result
            cidmul = arith.MulIOp(cid, sub_bnum).result
            vid = arith.AddIOp(cidmul, sub_bid).result
            num_blocks = pto.GetBlockNumOp().result

            # NOTE: convert i64, i32 all to index type for avoid type mismatch for arith ops
            vid_idx = arith.IndexCastOp(IndexType.get(), vid).result
            num_blocks_idx = arith.IndexCastOp(IndexType.get(), num_blocks).result
            total_elements = arith.IndexCastOp(IndexType.get(), argN).result

            elements_per_core = arith.DivUIOp(total_elements, num_blocks_idx).result
            num_tiles = arith.DivUIOp(elements_per_core, c1024).result

            vec_section = pto.SectionVectorOp()
            vec_block = vec_section.body.blocks.append()
            with InsertionPoint(vec_block):

                tv0 = pto.MakeTensorViewOp(tv1_f32, arg0, [total_elements], [c1]).result
                tv1 = pto.MakeTensorViewOp(tv1_f32, arg1, [total_elements], [c1]).result
                tv2 = pto.MakeTensorViewOp(tv1_f32, arg2, [total_elements], [c1]).result

                tb0 = pto.AllocTileOp(tile_buf).result
                tb1 = pto.AllocTileOp(tile_buf).result
                tb2 = pto.AllocTileOp(tile_buf).result

                # NOTE: `scf.for_` syntax sugar defined in https://github.com/llvm/llvm-project/blob/llvmorg-19.1.7/mlir/python/mlir/dialects/scf.py#L106
                for i in scf.for_(c0, num_tiles, c1):
                    offset_core = arith.MulIOp(vid_idx, elements_per_core).result
                    offset_tile = arith.MulIOp(i, c1024).result
                    offset_global = arith.AddIOp(offset_core, offset_tile).result

                    sv0 = pto.PartitionViewOp(
                        tile_view_32, tv0, offsets=[offset_global], sizes=[c1024]
                    ).result
                    sv1 = pto.PartitionViewOp(
                        tile_view_32, tv1, offsets=[offset_global], sizes=[c1024]
                    ).result
                    sv2 = pto.PartitionViewOp(
                        tile_view_32, tv2, offsets=[offset_global], sizes=[c1024]
                    ).result

                    pto.TLoadOp(None, sv0, tb0)
                    pto.TLoadOp(None, sv1, tb1)
                    pto.TAddOp(tb0, tb1, tb2)
                    pto.TStoreOp(None, tb2, sv2)

                    scf.YieldOp([])

            func.ReturnOp([])

        m.operation.verify()
        return m


if __name__ == "__main__":
    print(build())
