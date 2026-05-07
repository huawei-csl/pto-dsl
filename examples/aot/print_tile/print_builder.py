from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const


# common, reusable type declarations
dtype = pto.float32
index_dtype = pto.int32
ptr_type = pto.PtrType(dtype)
subtensor_shape = [32, 32]
tile_cfg = pto.TileBufConfig()
# defaults to pto.TileBufConfig(blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null")
tile_type = pto.TileBufType(
    shape=[32, 32],
    valid_shape=[-1, -1],
    dtype=dtype,
    memory_space="VEC",
    config=tile_cfg,
)


@to_ir_module
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

    v_row_idx = s.index_cast(arg_vrow_i32)
    v_col_idx = s.index_cast(arg_vcol_i32)

    tv0 = pto.as_tensor(ptr=arg0, shape=[c1280, c32], strides=[c32, c1])
    tv1 = pto.as_tensor(ptr=arg1, shape=[c1280, c32], strides=[c32, c1])
    tv2 = pto.as_tensor(ptr=arg2, shape=[c1280, c32], strides=[c32, c1])

    vid_idx = s.index_cast(vid)
    offset_row = vid_idx * c32  # every core loads 32 rows of data
    sv0 = pto.slice_view(source=tv0, offsets=[offset_row, c0], sizes=[c32, c32])
    sv1 = pto.slice_view(source=tv1, offsets=[offset_row, c0], sizes=[c32, c32])
    sv2 = pto.slice_view(source=tv2, offsets=[offset_row, c0], sizes=[c32, c32])

    with pto.vector_section():
        tb0 = pto.alloc_tile(tile_type, valid_row=v_row_idx, valid_col=v_col_idx)
        tb1 = pto.alloc_tile(tile_type, valid_row=v_row_idx, valid_col=v_col_idx)
        tb2 = pto.alloc_tile(tile_type, valid_row=v_row_idx, valid_col=v_col_idx)

        pto.load(sv0, tb0)
        pto.load(sv1, tb1)
        pto.print("hello%d\n", s.index_cast(c1, pto.int32))
        tile.print(tb0)
        tile.add(tb0, tb1, tb2)
        pto.store(tb2, sv2)


if __name__ == "__main__":
    module = vec_add_kernel_2d_dynamic
    print(module)
