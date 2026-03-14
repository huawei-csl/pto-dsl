import ctypes
import torch
import torch_npu
from ptodsl.test_util import get_test_device


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def lib_to_func(lib):
    def add_func(x, y, z, stream_ptr=None):

        vrow, vcol = 32, 32  # local tile shape hard-coded as the kernel

        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        lib.call_kernel(
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(z),
            vrow,
            vcol,
        )

    return add_func


def test_add():
    device = get_test_device()
    torch.npu.set_device(device)

    lib_path = "./print_lib.so"
    lib = ctypes.CDLL(lib_path)
    add_func = lib_to_func(lib)

    shape = [1280, 32]  # tensor shape hard-coded as the kernel
    torch.manual_seed(0)
    dtype = torch.float32
    x = torch.arange(shape[0] * shape[1], device=device, dtype=dtype).reshape(shape)
    y = torch.arange(shape[0] * shape[1], device=device, dtype=dtype).reshape(shape)
    z = torch.empty(shape, device=device, dtype=dtype)

    add_func(x, y, z)
    torch.npu.synchronize()


if __name__ == "__main__":
    test_add()
