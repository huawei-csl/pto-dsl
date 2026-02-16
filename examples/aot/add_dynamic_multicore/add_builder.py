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

        tensor_view = pto.TensorViewType.get(1, f32)

        tile_length = 1024  # TODO: increase to 8192 for better DMA util
        tile_view = pto.PartitionTensorViewType.get([1, tile_length], f32)
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
            c_tile = arith.ConstantOp(IndexType.get(), tile_length).result

            arg0, arg1, arg2, argN = entry.arguments

            cid = pto.GetBlockIdxOp().result
            sub_bid = pto.GetSubBlockIdxOp().result
            sub_bnum = pto.GetSubBlockNumOp().result
            cidmul = arith.MulIOp(cid, sub_bnum).result
            vid = arith.AddIOp(cidmul, sub_bid).result
            num_blocks = pto.GetBlockNumOp().result

            # NOTE: convert i64, i32 all to index type for avoid type mismatch for arith ops
            vid_idx = arith.IndexCastOp(IndexType.get(), vid).result
            num_cores = arith.IndexCastOp(IndexType.get(), num_blocks).result
            total_elements = arith.IndexCastOp(IndexType.get(), argN).result

            # https://mlir.llvm.org/docs/Dialects/ArithOps/#arithceildivsi-arithceildivsiop
            num_tiles_global = arith.CeilDivSIOp(total_elements, c_tile).result
            num_tiles_per_core = arith.CeilDivSIOp(num_tiles_global, num_cores).result
            tile_offset_this_core = arith.MulIOp(vid_idx, num_tiles_per_core).result

            vec_section = pto.SectionVectorOp()
            vec_block = vec_section.body.blocks.append()
            with InsertionPoint(vec_block):

                tv0 = pto.MakeTensorViewOp(tensor_view, arg0, [total_elements], [c1]).result
                tv1 = pto.MakeTensorViewOp(tensor_view, arg1, [total_elements], [c1]).result
                tv2 = pto.MakeTensorViewOp(tensor_view, arg2, [total_elements], [c1]).result

                tb0 = pto.AllocTileOp(tile_buf).result
                tb1 = pto.AllocTileOp(tile_buf).result
                tb2 = pto.AllocTileOp(tile_buf).result

                # Skip whole core if its starting tile is already out-of-bound.
                has_valid_start_tile = arith.CmpIOp(
                    arith.CmpIPredicate.slt, tile_offset_this_core, num_tiles_global
                ).result
                core_if = scf.IfOp(has_valid_start_tile)
                with InsertionPoint(core_if.then_block):
                    tiles_end_this_core = arith.AddIOp(
                        tile_offset_this_core, num_tiles_per_core
                    ).result
                    need_truncate = arith.CmpIOp(
                        arith.CmpIPredicate.sgt, tiles_end_this_core, num_tiles_global
                    ).result
                    remaining_tiles = arith.SubIOp(
                        num_tiles_global, tile_offset_this_core
                    ).result

                    # Truncate per-core tiles when the chunk crosses global bound.
                    tiles_to_process_if = scf.IfOp(
                        need_truncate, [IndexType.get()], hasElse=True
                    )
                    with InsertionPoint(tiles_to_process_if.then_block):
                        scf.YieldOp([remaining_tiles])
                    with InsertionPoint(tiles_to_process_if.else_block):
                        scf.YieldOp([num_tiles_per_core])
                    tiles_to_process = tiles_to_process_if.results[0]

                    elements_to_process = arith.MulIOp(tiles_to_process, c_tile).result
                    has_elements = arith.CmpIOp(
                        arith.CmpIPredicate.sgt, elements_to_process, c0
                    ).result
                    work_if = scf.IfOp(has_elements)
                    with InsertionPoint(work_if.then_block):
                        # NOTE: `scf.for_` syntax sugar defined in https://github.com/llvm/llvm-project/blob/llvmorg-19.1.7/mlir/python/mlir/dialects/scf.py#L106
                        for i in scf.for_(c0, tiles_to_process, c1):
                            tile_offset_global = arith.AddIOp(
                                i, tile_offset_this_core
                            ).result
                            offset_global = arith.MulIOp(tile_offset_global, c_tile).result

                            sv0 = pto.PartitionViewOp(
                                tile_view, tv0, offsets=[offset_global], sizes=[c_tile]
                            ).result
                            sv1 = pto.PartitionViewOp(
                                tile_view, tv1, offsets=[offset_global], sizes=[c_tile]
                            ).result
                            sv2 = pto.PartitionViewOp(
                                tile_view, tv2, offsets=[offset_global], sizes=[c_tile]
                            ).result

                            pto.TLoadOp(None, sv0, tb0)
                            pto.TLoadOp(None, sv1, tb1)
                            pto.TAddOp(tb0, tb1, tb2)
                            pto.TStoreOp(None, tb2, sv2)

                            scf.YieldOp([])
                        scf.YieldOp([])
                    scf.YieldOp([])

            func.ReturnOp([])

        m.operation.verify()
        return m


if __name__ == "__main__":
    print(build())
