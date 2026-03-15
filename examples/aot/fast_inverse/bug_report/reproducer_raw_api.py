from mlir.dialects import arith, func, pto as _pto
from mlir.ir import (
    Context,
    F16Type,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
)


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build_raw_module():
    with Context() as ctx, Location.unknown():
        _pto.register_dialect(ctx, load=True)
        module = Module.create()

        in_dtype = F16Type.get()
        out_dtype = F32Type.get()
        i32 = IntegerType.get_signless(32)

        ptr_in = _pto.PtrType.get(in_dtype)
        ptr_out = _pto.PtrType.get(out_dtype)

        tv_in_ty = _pto.TensorViewType.get(2, in_dtype)
        tv_out_ty = _pto.TensorViewType.get(2, out_dtype)
        st_in_ty = _pto.PartitionTensorViewType.get([128, 128], in_dtype)
        st_out_ty = _pto.PartitionTensorViewType.get([128, 128], out_dtype)

        mat = _pto.AddressSpaceAttr.get(_pto.AddressSpace.MAT)
        left = _pto.AddressSpaceAttr.get(_pto.AddressSpace.LEFT)
        right = _pto.AddressSpaceAttr.get(_pto.AddressSpace.RIGHT)
        acc = _pto.AddressSpaceAttr.get(_pto.AddressSpace.ACC)

        cfg_mat = _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.ColMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.RowMajor),
            _pto.TileConfig.fractalABSize,
            _pto.PadValueAttr.get(_pto.PadValue.Null),
        )
        cfg_left = _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.RowMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.RowMajor),
            _pto.TileConfig.fractalABSize,
            _pto.PadValueAttr.get(_pto.PadValue.Null),
        )
        cfg_right = _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.RowMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.ColMajor),
            _pto.TileConfig.fractalABSize,
            _pto.PadValueAttr.get(_pto.PadValue.Null),
        )
        cfg_acc = _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.ColMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.RowMajor),
            _pto.TileConfig.fractalCSize,
            _pto.PadValueAttr.get(_pto.PadValue.Null),
        )

        tile_mat = _pto.TileBufType.get([128, 128], in_dtype, mat, [-1, -1], cfg_mat)
        tile_left = _pto.TileBufType.get(
            [128, 128], in_dtype, left, [-1, -1], cfg_left
        )
        tile_right = _pto.TileBufType.get(
            [128, 128], in_dtype, right, [-1, -1], cfg_right
        )
        tile_acc = _pto.TileBufType.get([128, 128], out_dtype, acc, [-1, -1], cfg_acc)

        fn_ty = func.FunctionType.get([ptr_out, ptr_in, i32], [])
        with InsertionPoint(module.body):
            fn = func.FuncOp("reproducer_raw_api", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            out_ptr, in_ptr, n_i32 = entry.arguments
            n = arith.IndexCastOp(IndexType.get(), n_i32).result
            c0 = _idx_const(0)
            c1 = _idx_const(1)

            cube = _pto.SectionCubeOp()
            cube_block = cube.body.blocks.append()
            with InsertionPoint(cube_block):
                tv_in = _pto.MakeTensorViewOp(tv_in_ty, in_ptr, [n, n], [n, c1]).result
                tv_out = _pto.MakeTensorViewOp(
                    tv_out_ty, out_ptr, [n, n], [n, c1]
                ).result

                sv_in = _pto.PartitionViewOp(
                    st_in_ty, tv_in, offsets=[c0, c0], sizes=[n, n]
                ).result
                sv_out = _pto.PartitionViewOp(
                    st_out_ty, tv_out, offsets=[c0, c0], sizes=[n, n]
                ).result

                tmat = _pto.AllocTileOp(tile_mat, valid_row=n, valid_col=n).result
                ta = _pto.AllocTileOp(tile_left, valid_row=n, valid_col=n).result
                tb = _pto.AllocTileOp(tile_right, valid_row=n, valid_col=n).result
                tc = _pto.AllocTileOp(tile_acc, valid_row=n, valid_col=n).result

                _pto.TLoadOp(None, sv_in, tmat)
                _pto.TMovOp(None, tmat, ta)
                _pto.TMovOp(None, tmat, tb)
                _pto.TMatmulOp(None, ta, tb, tc)
                _pto.TMovOp(None, tc, tmat)  # ACC -> MAT: expected failure
                _pto.TStoreOp(None, tc, sv_out)

            func.ReturnOp([])

        module.operation.verify()
        return module


if __name__ == "__main__":
    print(build_raw_module())
