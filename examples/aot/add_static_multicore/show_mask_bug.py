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

        vrow, vcol = 32, 32  # local tile shape hard-coded as the kernel

        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        lib.call_kernel(
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(z),
            vrow, vcol
        )
    return add_func


def lib_to_func(lib):
    def add_func(
        x,
        y,
        z,
        stream_ptr=None
        ):

        vrow, vcol = 32, 32  # local tile shape hard-coded as the kernel

        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        lib.call_kernel(
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(z),
            vrow, vcol
        )
    return add_func


def mask_func(lib_path):
    lib = ctypes.CDLL(lib_path)
    default_stream = torch.npu.current_stream()._as_parameter_

    def call_good_mask(stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = default_stream
        lib.call_good_mask(stream_ptr)

    def call_bad_mask(stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = default_stream
        lib.call_bad_mask(stream_ptr)

    return call_good_mask, call_bad_mask


def test_add(show_bug):
    device = get_test_device()
    torch.npu.set_device(device)

    lib_path = "./add_lib.so"
    lib = ctypes.CDLL(lib_path)
    add_func = lib_to_func(lib)

    call_good_mask, call_bad_mask = mask_func("./mask_lib.so")

    shape = [1280, 32]  # tensor shape hard-coded as the kernel
    torch.manual_seed(0)
    dtype = torch.float32
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    repeat = 50
    if show_bug:
        for _ in range(repeat):
            call_bad_mask()
            add_func(x, y, z)
    else:
        for _ in range(repeat):
            call_good_mask()
            add_func(x, y, z)

    torch.npu.synchronize()

    z_ref = x + y
    torch.testing.assert_close(z, z_ref)
    print("result equal!")

if __name__ == "__main__":
    # test_add(show_bug=False)
    test_add(show_bug=True)
