import argparse
import ctypes
import math
import random
from typing import Callable

import numpy as np
import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device

SUPPORTED_SIZES = (16, 32, 64, 96, 128)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # m
        ctypes.c_void_p,  # i_neg
        ctypes.c_uint32,  # matrix_size
        ctypes.c_uint32,  # max_block_size
    ]
    lib.call_kernel.restype = None

    def tri_inv_func(out, m, i_neg, matrix_size, max_block_size, block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(out),
            torch_to_ctypes(m),
            torch_to_ctypes(i_neg),
            ctypes.c_uint32(matrix_size),
            ctypes.c_uint32(max_block_size),
        )

    return tri_inv_func


def block_ones_matrix(n, block_dim_x, block_dim_y):
    u_block = np.ones((16, 16))
    n_blocks = n // 16
    u = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(n_blocks):
                start = i * 16
                end = start + 16
                u[x, y, start:end, start:end] = u_block
    return torch.from_numpy(np.triu(u, 1))


def block_random_matrix(n, block_dim_x, block_dim_y, scale=0.2):
    u_block = scale * np.random.rand(16, 16)
    u_block = np.triu(u_block, k=1)
    u = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                u[x, y, i : i + 16, i : i + 16] = u_block.copy()
    return torch.from_numpy(u)


def _next_pow2(n):
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def tri_inv_trick(tri_inv_func, u, max_block_size=None):
    n = u.shape[-1]
    if n not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported matrix size {n}. Supported sizes: {SUPPORTED_SIZES}.")

    u_half = u.to(torch.float16)
    flat = u_half.reshape(-1, n, n).contiguous()
    block_dim = flat.shape[0]
    out = torch.empty((block_dim, n, n), device=flat.device, dtype=torch.float32)

    i_neg = -torch.eye(n, device=flat.device, dtype=torch.float16).contiguous()
    if max_block_size is None:
        max_block_size = _next_pow2(n)

    tri_inv_func(out, flat, i_neg, n, max_block_size, block_dim)
    return out.reshape(*u.shape[:-2], n, n)


def _test_tri_inv_trick(
    tri_inv_func,
    u: torch.Tensor,
    atol: float,
    rtol: float,
    ftol: float,
    max_block_size=None,
):
    n = u.shape[-1]
    u = u.to(torch.float16)
    u_npu = u.npu()
    torch.npu.synchronize()

    identity = np.eye(n, dtype=np.double)
    golden_numpy = np.zeros(u.shape, dtype=np.double)
    u_cpu = u.cpu()
    for x in range(u.shape[0]):
        for y in range(u.shape[1]):
            golden_numpy[x, y] = np.linalg.inv(u_cpu[x, y].numpy().astype(np.double) + identity)
    golden_cpu = torch.from_numpy(golden_numpy)

    torch.npu.synchronize()
    actual = tri_inv_trick(tri_inv_func, u_npu, max_block_size=max_block_size)
    torch.npu.synchronize()
    actual_cpu = actual.cpu().to(torch.float64)

    frob_error = torch.sqrt(
        torch.sum((golden_cpu - actual_cpu) * (golden_cpu - actual_cpu))
        / torch.sum(golden_cpu * golden_cpu)
    )
    actual_numpy = actual_cpu.numpy()
    golden_numpy = golden_cpu.numpy()

    assert np.allclose(
        actual_numpy, golden_numpy, atol=atol, rtol=rtol
    ), f"Error at allclose - tensor shape: {u.shape} - rtol: {rtol}."
    assert frob_error <= ftol, f"frob_error: {frob_error}"


def run_all_tests(tri_inv_func, max_block_size=None):
    matrix_generators: list[tuple[Callable, float, float, float]] = [
        (block_ones_matrix, 0.0, 0.0, 0.0),
        (block_random_matrix, 5e-5, 0.1, 1e-4),
    ]
    for n in SUPPORTED_SIZES:
        for block_dim_x in (1, 3, 7, 16):
            for block_dim_y in (1, 2, 4, 16):
                for matrix_gen, atol, rtol, ftol in matrix_generators:
                    u = matrix_gen(n, block_dim_x, block_dim_y)
                    _test_tri_inv_trick(
                        tri_inv_func,
                        u,
                        atol=atol,
                        rtol=rtol,
                        ftol=ftol,
                        max_block_size=max_block_size,
                    )
                    print(
                        f"[ok] n={n}, bx={block_dim_x}, by={block_dim_y}, generator={matrix_gen.__name__}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Use manual-sync library instead of the default auto-sync library.",
    )
    parser.add_argument(
        "--max-block-size",
        type=int,
        default=None,
        help="Override kernel max_block_size (default: next power of two of n).",
    )
    args = parser.parse_args()

    lib_path = (
        "./tri_inv_trick_manual_sync_lib.so"
        if args.manual_sync
        else "./tri_inv_trick_auto_sync_lib.so"
    )

    device = get_test_device()
    torch.npu.set_device(device)

    tri_inv_func = load_lib(lib_path=lib_path)
    run_all_tests(tri_inv_func, max_block_size=args.max_block_size)
