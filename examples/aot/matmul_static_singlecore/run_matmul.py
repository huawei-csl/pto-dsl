import ctypes
import torch
import torch_npu
from ptodsl.test_util import get_test_device


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)

    default_block_dim = 1  # NOTE: kernel is single-core for now

    def matmul_func(
        c, a, b,
        block_dim=default_block_dim,
        stream_ptr=None
    ):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(c),
            torch_to_ctypes(a),
            torch_to_ctypes(b),
        )

    return matmul_func


def test_matmul():
    device = get_test_device()
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    m, k, n = 32, 256, 32
    torch.manual_seed(0)
    a = torch.rand((m,k), device=device, dtype=dtype)
    b = torch.rand((k,n), device=device, dtype=dtype)
    c = torch.zeros((m, n), device=device, dtype=dtype)

    matmul_func = load_lib("./matmul_kernel.so")
    matmul_func(c, a, b)
    torch.npu.synchronize()

    c_ref = torch.matmul(a, b)
    diff = (c - c_ref).abs().max()
    print('max diff: ', diff)


if __name__ == "__main__":
    test_matmul()
