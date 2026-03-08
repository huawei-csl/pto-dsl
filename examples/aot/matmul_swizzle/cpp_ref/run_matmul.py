import os
import ctypes
import torch
import torch.nn.functional as F
import torch_npu
from ptodsl.test_util import get_test_device


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    # call_kernel(blockDim, stream, x, y, z, M, N, K)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x [M, K]
        ctypes.c_void_p,  # y [N, K]
        ctypes.c_void_p,  # z [M, N]
        ctypes.c_int,  # M
        ctypes.c_int,  # N
        ctypes.c_int,  # K
    ]
    lib.call_kernel.restype = None

    def _matmul_single(a, b, block_dim, stream_ptr):
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

    DEFAULT_BLOCK_DIM = 24

    def matmul_abt(
        a,
        b,
        block_dim=DEFAULT_BLOCK_DIM,
        stream_ptr=None,
    ):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul_abt expects 2D tensors: a[M,K], b[N,K]")
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"K mismatch: a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
            )
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise ValueError("matmul_abt currently supports float16 inputs only")

        if stream_ptr is None:
            stream = torch.npu.current_stream()._as_parameter_

        return _matmul_single(a, b, block_dim, stream_ptr)

    return matmul_abt


def test_matmul():
    device = "npu:7"
    torch.npu.set_device(device)

    matmul_abt = load_lib("./matmul_kernel.so")

    BLOCK_DIM_LIST = [1, 20, 24]
    M_LIST = [128 * i for i in range(1, 37, 4)]  # 128, ..., 4096
    SHAPES_NK = [
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
    ]
    dtype = torch.float16

    torch.manual_seed(0)

    for m in M_LIST:
        for n, k in SHAPES_NK:
            a = torch.randn(m, k, dtype=dtype, device="npu")
            b = torch.randn(n, k, dtype=dtype, device="npu")
            c_ref = F.linear(a, b)
            torch.npu.synchronize()

            for block_dim in BLOCK_DIM_LIST:
                c = matmul_abt(a, b, block_dim=block_dim)
                torch.npu.synchronize()
                # torch.testing.assert_close(c, c_ref, atol=1e-4, rtol=1e-2)
                absdiff = torch.mean(torch.abs(c - c_ref))
                print(f"(m, n, k, block_dim) = ({m}, {n}, {k}, {block_dim}): absdiff {absdiff}")

if __name__ == "__main__":
    test_matmul()
