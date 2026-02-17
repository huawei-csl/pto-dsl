from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto, scf
from mlir.ir import F32Type, IndexType, IntegerType

def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            # The "tiles" are [1, tile_w]
            tile_w = 32

            f32 = F32Type.get(ctx)
            u32 = IntegerType.get_signless(32, ctx)
            idx = IndexType.get(ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv1_f32 = pto.TensorViewType.get(1, f32, ctx)
            tile_view = pto.PartitionTensorViewType.get([tile_w], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            # Dynamic valid shape so we can mask partial tiles via valid_row/valid_col.
            tile_buf = pto.TileBufType.get([1, tile_w], f32, vec, [-1, -1], cfg, ctx)

            # function signature: (float*, float*, uint32 N)
            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, u32], [])

            with InsertionPoint(m.body):
                fn = func.FuncOp("sync_kernel_dyn", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                vec_section = pto.SectionVectorOp()
                vec_block = vec_section.body.blocks.append()
                
                with InsertionPoint(vec_block):
                    arg0, arg1, argN = entry.arguments
                    c0 = arith.ConstantOp(idx, 0).result
                    c1 = arith.ConstantOp(idx, 1).result
                    c_tile_w = arith.ConstantOp(idx, tile_w).result
                    total_elements = arith.IndexCastOp(idx, argN).result

                    num_blocks = arith.IndexCastOp(idx, pto.GetBlockNumOp()).result
                    num_el_per_core = arith.CeilDivSIOp(total_elements, num_blocks).result

                    # Per-core range: [core_start, core_end)
                    bid_raw = pto.GetBlockIdxOp().result
                    bid = arith.IndexCastOp(idx, bid_raw).result
                    core_start = arith.MulIOp(bid, num_el_per_core).result
                    core_end_unclamped = arith.AddIOp(core_start, num_el_per_core).result
                    core_end = arith.MinUIOp(core_end_unclamped, total_elements).result
                    core_len = arith.SubIOp(core_end, core_start).result

                    # Per-core number of tiles: ceil(core_len / tile_w).
                    num_tiles = arith.CeilDivSIOp(core_len, c_tile_w).result

                    # GM tensors shape N with stride 1
                    tv0 = pto.MakeTensorViewOp(tv1_f32, arg0, [total_elements], [c1]).result
                    tv1 = pto.MakeTensorViewOp(tv1_f32, arg1, [total_elements], [c1]).result

                    # for loop: for i in range(num_tiles)
                    loop = scf.ForOp(c0, num_tiles, c1)
                    with InsertionPoint(loop.body):
                        i = loop.induction_variable
                        offset_tile = arith.MulIOp(i, c_tile_w).result
                        offset_total = arith.AddIOp(core_start, offset_tile).result

                        remaining_core = arith.SubIOp(core_end, offset_total).result
                        valid_len = arith.MinUIOp(remaining_core, c_tile_w).result

                        # TODO: shouldnt allocate a new tile in UB for every iteration?
                        # https://github.com/zhangstevenunity/PTOAS/issues/111
                        tb0 = pto.AllocTileOp(tile_buf, valid_row=c1, valid_col=valid_len).result
                        tb1 = pto.AllocTileOp(tile_buf, valid_row=c1, valid_col=valid_len).result

                        # each core c takes a tile at offset c*nun_el_per_core+i*tile_w  
                        sv0 = pto.PartitionViewOp(
                            tile_view,
                            tv0,
                            offsets=[offset_total],
                            sizes=[c_tile_w]
                        ).result
                        sv1 = pto.PartitionViewOp(
                            tile_view,
                            tv1,
                            offsets=[offset_total],
                            sizes=[c_tile_w]
                        ).result

                        pto.TLoadOp(None, sv0, tb0)
                        pto.TReluOp(tb0, tb1)
                        pto.TStoreOp(None, tb1, sv1)

                        scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m

if __name__ == "__main__":
    print(build())
