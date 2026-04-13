from ptodsl import jit, pto, tile
from ptodsl import scalar as s
import torch
import torch_npu
from ptodsl.npu_info import get_num_cube_cores, get_test_device

_BLOCK_DIM = get_num_cube_cores()
_SUBBLOCK_NUM = 2  # 910B2 and 910B4 both have 2 subblocks per cube core
_ROWS_PER_CORE = 32
_TOTAL_ROWS = _BLOCK_DIM * _SUBBLOCK_NUM * _ROWS_PER_CORE

const = s.const


def meta_data():
    # common, reusable type declarations
    dtype = pto.float32
    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=2, dtype=dtype)
    subtensor_type = pto.SubTensorType(
        shape=[32, 32], dtype=dtype
    )  # TODO: omit shape https://github.com/zhangstevenunity/PTOAS/issues/31
    tile_cfg = pto.TileBufConfig()
    # defaults to pto.TileBufConfig(blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null")
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


@jit(meta_data=meta_data, block_dim=_BLOCK_DIM)
def vec_add_kernel(
    arg0: "ptr_type",
    arg1: "ptr_type",
    arg2: "ptr_type",
    vrow: "index_dtype",
    vcol: "index_dtype",
) -> None:
    c0 = const(0)
    c1 = const(1)
    c32 = const(_ROWS_PER_CORE)
    c1280 = const(_TOTAL_ROWS)  # rows/core * num_cores * subblocks_per_core

    cid = pto.get_block_idx()
    sub_bid = pto.get_subblock_idx()
    sub_bnum = pto.get_subblock_num()
    cidmul = cid * sub_bnum
    vid = cidmul + sub_bid

    v_row_idx = s.index_cast(vrow)
    v_col_idx = s.index_cast(vcol)

    tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[c1280, c32], strides=[c32, c1])
    tv1 = pto.as_tensor(tensor_type, ptr=arg1, shape=[c1280, c32], strides=[c32, c1])
    tv2 = pto.as_tensor(tensor_type, ptr=arg2, shape=[c1280, c32], strides=[c32, c1])

    vid_idx = s.index_cast(vid)
    offset_row = vid_idx * c32  # every core loads 32 rows of data
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
        tile.add(tb0, tb1, tb2)
        pto.store(tb2, sv2)


def test_add():
    device = get_test_device()
    torch.npu.set_device(device)

    shape = [
        _TOTAL_ROWS,
        _ROWS_PER_CORE,
    ]  # total_rows = block_dim * subblocks * rows_per_core
    torch.manual_seed(0)
    dtype = torch.float32
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    vec_add_kernel(x, y, z, 32, 32)
    torch.npu.synchronize()

    z_ref = x + y
    torch.testing.assert_close(z, z_ref)
    print("result equal!")


if __name__ == "__main__":
    test_add()
