from mlir.ir import Context, F32Type, InsertionPoint, IntegerType, Location, Module
from mlir.ir import IndexType
from mlir.dialects import arith, func, pto as _pto, scf

from ptodsl import to_ir_module

import ptodsl.language as pto

const = pto.const


def meta_data():
    dtype = pto.float32
    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=1, dtype=dtype)
    tile_length = 1024
    subtensor_type = pto.SubTensorType(shape=[1, tile_length], dtype=dtype)
    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[1, tile_length],
        valid_shape=[1, tile_length],
        dtype=dtype,
        memory_space="VEC",
        config=tile_cfg,
    )
    return {
        "ptr_type": ptr_type,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_type": subtensor_type,
        "tile_type": tile_type,
        "tile_length": tile_length,
    }


def vec_add_1d_dynamic(
    arg0: "ptr_type",
    arg1: "ptr_type",
    arg2: "ptr_type",
    argN: "index_dtype",
) -> None:
    c0 = const(0)
    c1 = const(1)
    c_tile = const(tile_length)

    cid = pto.get_block_idx()
    sub_bid = pto.get_subblock_idx()
    sub_bnum = pto.get_subblock_num()
    cidmul = cid * sub_bnum
    vid = cidmul + sub_bid
    num_blocks = pto.get_block_num()

    vid_idx = pto.index_cast(vid)
    num_cores = pto.index_cast(num_blocks)
    total_elements = pto.index_cast(argN)

    num_tiles_global = pto.ceil_div(total_elements, c_tile)
    num_tiles_per_core = pto.ceil_div(num_tiles_global, num_cores)
    tile_offset_this_core = vid_idx * num_tiles_per_core

    with pto.vector_section():
        tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[total_elements], strides=[c1])
        tv1 = pto.as_tensor(tensor_type, ptr=arg1, shape=[total_elements], strides=[c1])
        tv2 = pto.as_tensor(tensor_type, ptr=arg2, shape=[total_elements], strides=[c1])

        tb0 = pto.alloc_tile(tile_type)
        tb1 = pto.alloc_tile(tile_type)
        tb2 = pto.alloc_tile(tile_type)

        with pto.if_context(tile_offset_this_core < num_tiles_global):
            tiles_end_this_core = tile_offset_this_core + num_tiles_per_core
            need_truncate = tiles_end_this_core > num_tiles_global
            remaining_tiles = num_tiles_global - tile_offset_this_core

            tiles_to_process = pto.select(
                need_truncate, remaining_tiles, num_tiles_per_core
            )
            elements_to_process = tiles_to_process * c_tile

            with pto.if_context(elements_to_process > c0):
                for i in pto.for_range(c0, tiles_to_process, c1):
                    tile_offset_global = i + tile_offset_this_core
                    offset_global = tile_offset_global * c_tile

                    sv0 = pto.slice_view(
                        subtensor_type, source=tv0, offsets=[offset_global], sizes=[c_tile]
                    )
                    sv1 = pto.slice_view(
                        subtensor_type, source=tv1, offsets=[offset_global], sizes=[c_tile]
                    )
                    sv2 = pto.slice_view(
                        subtensor_type, source=tv2, offsets=[offset_global], sizes=[c_tile]
                    )

                    pto.load(sv0, tb0)
                    pto.load(sv1, tb1)
                    pto.add(tb0, tb1, tb2)
                    pto.store(tb2, sv2)


def build_verbose():
    with Context() as ctx, Location.unknown():
        _pto.register_dialect(ctx, load=True)

        module = Module.create()

        f32 = F32Type.get()
        i32 = IntegerType.get_signless(32)
        ptr_f32 = _pto.PtrType.get(f32)

        tensor_view = _pto.TensorViewType.get(1, f32)
        tile_length = 1024
        tile_view = _pto.PartitionTensorViewType.get([1, tile_length], f32)

        vec = _pto.AddressSpaceAttr.get(_pto.AddressSpace.VEC)
        bl = _pto.BLayoutAttr.get(_pto.BLayout.RowMajor)
        sl = _pto.SLayoutAttr.get(_pto.SLayout.NoneBox)
        pd = _pto.PadValueAttr.get(_pto.PadValue.Null)
        cfg = _pto.TileBufConfigAttr.get(bl, sl, 512, pd)
        tile_buf = _pto.TileBufType.get([1, tile_length], f32, vec, [1, tile_length], cfg)
        fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32, i32], [])

        with InsertionPoint(module.body):
            fn = func.FuncOp("vec_add_1d_dynamic", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            c0 = arith.ConstantOp(IndexType.get(), 0).result
            c1 = arith.ConstantOp(IndexType.get(), 1).result
            c_tile = arith.ConstantOp(IndexType.get(), tile_length).result

            arg0, arg1, arg2, argN = entry.arguments

            cid = _pto.GetBlockIdxOp().result
            sub_bid = _pto.GetSubBlockIdxOp().result
            sub_bnum = _pto.GetSubBlockNumOp().result
            cidmul = arith.MulIOp(cid, sub_bnum).result
            vid = arith.AddIOp(cidmul, sub_bid).result
            num_blocks = _pto.GetBlockNumOp().result

            vid_idx = arith.IndexCastOp(IndexType.get(), vid).result
            num_cores = arith.IndexCastOp(IndexType.get(), num_blocks).result
            total_elements = arith.IndexCastOp(IndexType.get(), argN).result

            num_tiles_global = arith.CeilDivSIOp(total_elements, c_tile).result
            num_tiles_per_core = arith.CeilDivSIOp(num_tiles_global, num_cores).result
            tile_offset_this_core = arith.MulIOp(vid_idx, num_tiles_per_core).result

            vec_section = _pto.SectionVectorOp()
            vec_block = vec_section.body.blocks.append()
            with InsertionPoint(vec_block):
                tv0 = _pto.MakeTensorViewOp(tensor_view, arg0, [total_elements], [c1]).result
                tv1 = _pto.MakeTensorViewOp(tensor_view, arg1, [total_elements], [c1]).result
                tv2 = _pto.MakeTensorViewOp(tensor_view, arg2, [total_elements], [c1]).result

                tb0 = _pto.AllocTileOp(tile_buf).result
                tb1 = _pto.AllocTileOp(tile_buf).result
                tb2 = _pto.AllocTileOp(tile_buf).result

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

                    tiles_to_process = arith.SelectOp(
                        need_truncate, remaining_tiles, num_tiles_per_core
                    ).result

                    elements_to_process = arith.MulIOp(tiles_to_process, c_tile).result
                    has_elements = arith.CmpIOp(
                        arith.CmpIPredicate.sgt, elements_to_process, c0
                    ).result
                    work_if = scf.IfOp(has_elements)
                    with InsertionPoint(work_if.then_block):
                        for i in scf.for_(c0, tiles_to_process, c1):
                            tile_offset_global = arith.AddIOp(i, tile_offset_this_core).result
                            offset_global = arith.MulIOp(tile_offset_global, c_tile).result

                            sv0 = _pto.PartitionViewOp(
                                tile_view, tv0, offsets=[offset_global], sizes=[c_tile]
                            ).result
                            sv1 = _pto.PartitionViewOp(
                                tile_view, tv1, offsets=[offset_global], sizes=[c_tile]
                            ).result
                            sv2 = _pto.PartitionViewOp(
                                tile_view, tv2, offsets=[offset_global], sizes=[c_tile]
                            ).result

                            _pto.TLoadOp(None, sv0, tb0)
                            _pto.TLoadOp(None, sv1, tb1)
                            _pto.TAddOp(tb0, tb1, tb2)
                            _pto.TStoreOp(None, tb2, sv2)

                            scf.YieldOp([])
                        scf.YieldOp([])
                    scf.YieldOp([])

            func.ReturnOp([])

        module.operation.verify()
        return module


def test_structural_ir_equality():
    dsl_module = to_ir_module(meta_data=meta_data)(vec_add_1d_dynamic)
    verbose_module = build_verbose()
    assert str(dsl_module) == str(verbose_module)
