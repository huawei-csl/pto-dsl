import ctypes
import os

import torch
import torch.nn.functional as F
import torch_npu

from ptodsl.test_util import get_test_device


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.call_kernel.restype = None

    def matmul_abt(a, b, *, block_dim=24, stream_ptr=None):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul_abt expects 2D tensors: a[M,K], b[N,K]")
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"K mismatch: a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
            )
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise ValueError("matmul_abt currently supports float16 inputs only")

        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        m = int(a.shape[0])
        k = int(a.shape[1])
        n = int(b.shape[0])
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)

        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            torch_to_ctypes(c),
            m,
            n,
            k,
        )
        return c

    return matmul_abt


def test_matmul():
    device = get_test_device()
    torch.npu.set_device(device)

    matmul_abt = load_lib("./matmul_kernel.so")

    block_dims = [1, 20, 24]
    shapes = [
        (128, 4096, 4096),
        (640, 8192, 8192),
        (1152, 4096, 16384),
    ]

    torch.manual_seed(0)
    for m, n, k in shapes:
        a = torch.randn(m, k, dtype=torch.float16, device=device)
        b = torch.randn(n, k, dtype=torch.float16, device=device)
        c_ref = F.linear(a, b)
        torch.npu.synchronize()

        for block_dim in block_dims:
            c = matmul_abt(a, b, block_dim=block_dim)
            torch.npu.synchronize()
            max_absdiff = (c - c_ref).abs().max().item()
            mean_absdiff = (c - c_ref).abs().mean().item()
            print(
                f"(m, n, k, block_dim)=({m}, {n}, {k}, {block_dim}) "
                f"max_absdiff={max_absdiff:.6f} mean_absdiff={mean_absdiff:.6f}"
            )


if __name__ == "__main__":
    test_matmul()
