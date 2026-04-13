import ctypes

import torch
import torch_npu  # noqa: F401

from ptodsl.npu_info import get_num_cube_cores, get_test_device

_DEFAULT_NUM_CORES = get_num_cube_cores()


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, block_dim=_DEFAULT_NUM_CORES):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # y
        ctypes.c_void_p,  # z
        ctypes.c_int32,  # N
    ]
    lib.call_kernel.restype = None

    def add_func(x, y, z, block_dim=block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        N = x.numel()
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(z),
            N,
        )

    return add_func


def test_add(lib_path="./add_lib.so", block_dim=_DEFAULT_NUM_CORES):
    device = get_test_device()
    torch.npu.set_device(device)

    add_func = load_lib(lib_path=lib_path, block_dim=block_dim)

    tile_size = 1024
    # Keep shapes aligned to tile size, but vary tile counts so they are not
    # required to be multiples of `block_dim`.
    tile_counts = [
        1,
        7,
        block_dim - 1,
        block_dim + 3,
        2 * block_dim + 7,
        5 * block_dim - 5,
    ]
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
    test_add("./add_double_lib.so")
