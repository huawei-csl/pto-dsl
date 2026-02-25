import ctypes
import torch
import torch_npu
from ptodsl.test_util import get_test_device


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def lib_to_func(lib):
    def add_func(
        x,
        y,
        z,
        stream_ptr=None
        ):

        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        N = x.numel()
        lib.call_kernel(
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(z),
            N
        )
    return add_func


def test_add():
    device = get_test_device()
    torch.npu.set_device(device)

    lib_path = "./add_lib.so"
    lib = ctypes.CDLL(lib_path)
    add_func = lib_to_func(lib)

    # shape parameter hard-coded as kernel
    num_cores = 24 * 2
    tile_size = 1024
    # Keep shapes aligned to tile size, but vary tile counts so they are not
    # required to be multiples of `num_cores`.
    tile_counts = [1, 7, num_cores - 1, num_cores + 3, 2 * num_cores + 7, 5 * num_cores - 5]
    shape_list = [tile_size * tiles for tiles in tile_counts]

    torch.manual_seed(0)
    dtype = torch.float32

    for shape in shape_list:
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)

        add_func(x, y, z)
        torch.npu.synchronize()

        z_ref = x + y
        torch.testing.assert_close(z, z_ref)
        print(f"result equal for shape {shape}")

if __name__ == "__main__":
    test_add()
