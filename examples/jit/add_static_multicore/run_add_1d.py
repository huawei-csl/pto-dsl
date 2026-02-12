from ptodsl import jit
import ptodsl.language as pto
import torch
import torch_npu

const = pto.const


def meta_data():
    # common, reusable type declarations
    dtype = pto.float32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=2, dtype=dtype)

    subtensor_type = pto.SubTensorType(shape=[1, 1024], dtype=dtype) 
    tile_type = pto.TileBufType(
        shape=[1, 1024], valid_shape=[-1, -1], dtype=dtype, memory_space="VEC", config=pto.TileBufConfig())

    return {
        "ptr_type": ptr_type,
        "tensor_type": tensor_type,
        "subtensor_type": subtensor_type,
        "tile_type": tile_type,
    }


@jit(meta_data=meta_data, block_dim=20)
def vec_add_kernel(
    arg0: "ptr_type",
    arg1: "ptr_type",
    arg2: "ptr_type"
    ) -> None:
    c0 = const(0)
    c1 = const(1)
    c1024 = const(1024)
    c20480 = const(20480)  # 1024 elements / core * 20 cores = 20480 elements

    cid = pto.get_block_idx()
    sub_bid = pto.get_subblock_idx()
    sub_bnum = pto.get_subblock_num()
    cidmul = cid * sub_bnum
    vid = cidmul + sub_bid

    tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[c1, c1024], strides=[c1024, c1])
    tv1 = pto.as_tensor(tensor_type, ptr=arg1, shape=[c1, c1024], strides=[c1024, c1])
    tv2 = pto.as_tensor(tensor_type, ptr=arg2, shape=[c1, c1024], strides=[c1024, c1])

    vid_idx = pto.index_cast(vid)
    offset = vid_idx * c1024  # every core loads 1024 elements of data
    sv0 = pto.slice_view(subtensor_type, source=tv0, offsets=[c0, offset], sizes=[c1, c1024])
    sv1 = pto.slice_view(subtensor_type, source=tv1, offsets=[c0, offset], sizes=[c1, c1024])
    sv2 = pto.slice_view(subtensor_type, source=tv2, offsets=[c0, offset], sizes=[c1, c1024])

    with pto.vector_section():
        tb0 = pto.alloc_tile(tile_type, valid_row=c1, valid_col=c1024)
        tb1 = pto.alloc_tile(tile_type, valid_row=c1, valid_col=c1024)
        tb2 = pto.alloc_tile(tile_type, valid_row=c1, valid_col=c1024)

        pto.load(sv0, tb0)
        pto.load(sv1, tb1)
        pto.add(tb0, tb1, tb2)
        pto.store(tb2, sv2)


def test_add():
    device = "npu:1"
    torch.npu.set_device(device)

    shape = (1, 1024 * 20)  # tensor shape hard-coded as the kernel
    torch.manual_seed(0)
    dtype = torch.float32
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    vec_add_kernel(x, y, z)
    torch.npu.synchronize()

    z_ref = x + y
    torch.testing.assert_close(z, z_ref)
    print("result equal!")

if __name__ == "__main__":
    test_add()
