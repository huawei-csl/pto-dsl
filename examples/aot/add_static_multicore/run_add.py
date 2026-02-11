import ctypes
import torch
import torch_npu


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


def test_add():
    device = "npu:0"
    torch.npu.set_device(device)

    lib_path = "./add_lib.so"
    lib = ctypes.CDLL(lib_path)
    add_func = lib_to_func(lib)

    shape = [1280, 32]  # tensor shape hard-coded as the kernel
    torch.manual_seed(0)
    dtype = torch.float32
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    add_func(x, y, z)
    torch.npu.synchronize()

    z_ref = x + y
    torch.testing.assert_close(z, z_ref)
    print("result equal!")

if __name__ == "__main__":
    test_add()
