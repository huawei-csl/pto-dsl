from ptodsl import to_ir_module
from ptodsl import pto, tile
from ptodsl import scalar as s

const = s.const


def meta_data():
    dtype = pto.float32
    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=2, dtype=dtype)
    subtensor_type = pto.SubTensorType(shape=[32, 32], dtype=dtype)
    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[32, 32],
        valid_shape=[-1, -1],
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
    }


def vec_sub_2d_static(
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

    tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[c1280, c32], strides=[c32, c1])
    tv1 = pto.as_tensor(tensor_type, ptr=arg1, shape=[c1280, c32], strides=[c32, c1])
    tv2 = pto.as_tensor(tensor_type, ptr=arg2, shape=[c1280, c32], strides=[c32, c1])

    vid_idx = s.index_cast(vid)
    offset_row = vid_idx * c32
    sv0 = pto.slice_view(
        subtensor_type, source=tv0, offsets=[offset_row, c0], sizes=[c32, c32]
    )
    sv1 = pto.slice_view(
        subtensor_type, source=tv1, offsets=[offset_row, c0], sizes=[c32, c32]
    )
    sv2 = pto.slice_view(
        subtensor_type, source=tv2, offsets=[offset_row, c0], sizes=[c32, c32]
    )

    with pto.vector_section():
        tb0 = pto.alloc_tile(tile_type, valid_row=v_row_idx, valid_col=v_col_idx)
        tb1 = pto.alloc_tile(tile_type, valid_row=v_row_idx, valid_col=v_col_idx)
        tb2 = pto.alloc_tile(tile_type, valid_row=v_row_idx, valid_col=v_col_idx)

        pto.load(sv0, tb0)
        pto.load(sv1, tb1)
        tile.sub(tb0, tb1, tb2)
        pto.store(tb2, sv2)


def test_ir_generation():
    ir_module = to_ir_module(meta_data=meta_data)(vec_sub_2d_static)
    ir_text = str(ir_module)
    assert "vec_sub_2d_static" in ir_text
    assert "pto.section.vector" in ir_text
    assert "pto.tload" in ir_text
    assert "pto.tsub" in ir_text
    assert "pto.tstore" in ir_text
