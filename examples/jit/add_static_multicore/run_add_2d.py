from ptodsl import jit
import ptodsl.language as pto
import torch
import torch_npu

const = pto.const


def meta_data():
    # common, reusable type declarations
    dtype = pto.float32
    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=2, dtype=dtype)
    subtensor_type = pto.SubTensorType(shape=[32, 32], dtype=dtype)  # TODO: omit shape https://github.com/zhangstevenunity/PTOAS/issues/31
    tile_cfg = pto.TileBufConfig()
    # defaults to pto.TileBufConfig(blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null")
    tile_type = pto.TileBufType(
        shape=[32, 32], valid_shape=[-1, -1], dtype=dtype, memory_space="VEC", config=tile_cfg)
    return {
        "ptr_type": ptr_type,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_type": subtensor_type,
        "tile_type": tile_type,
    }


@jit(meta_data=meta_data, block_dim=20)
def vec_add_kernel(
    arg0: "ptr_type",
    arg1: "ptr_type",
    arg2: "ptr_type",
    vrow: "index_dtype",
    vcol: "index_dtype"
    ) -> None:
    c0 = const(0)
    c1 = const(1)
    c32 = const(32)
    c1280 = const(1280)  # 32 rows/per * 40 vec cores = 1280 rows

    cid = pto.get_block_idx()
    sub_bid = pto.get_subblock_idx()
    sub_bnum = pto.get_subblock_num()
    cidmul = cid * sub_bnum
    vid = cidmul + sub_bid

    v_row_idx = pto.index_cast(vrow)
    v_col_idx = pto.index_cast(vcol)

    tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[c1280, c32], strides=[c32, c1])
    tv1 = pto.as_tensor(tensor_type, ptr=arg1, shape=[c1280, c32], strides=[c32, c1])
    tv2 = pto.as_tensor(tensor_type, ptr=arg2, shape=[c1280, c32], strides=[c32, c1])

    vid_idx = pto.index_cast(vid)
    offset_row = vid_idx * c32  # every core loads 32 rows of data
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


def test_add():
    device = "npu:1"
    torch.npu.set_device(device)

    shape = [1280, 32]  # tensor shape hard-coded as the kernel
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
