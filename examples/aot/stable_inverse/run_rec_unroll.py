import argparse
import ctypes
import random
from typing import Callable

import numpy as np
import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device

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
        ctypes.c_void_p,  # in
        ctypes.c_void_p,  # identity_neg
        ctypes.c_uint32,  # matrix_size
        ctypes.c_uint32,  # num_matrices
        ctypes.c_uint32,  # num_bsnd_heads
    ]
    lib.call_kernel.restype = None
    return lib


def random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    return scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)


def ones_triu_matrix(n, block_dim_x, block_dim_y):
    return torch.triu(torch.ones((block_dim_x, block_dim_y, n, n)), diagonal=1)


def block_ones_triu_matrix(n, block_dim_x, block_dim_y):
    block = np.ones((16, 16))
    n_blocks = n // 16
    out = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(n_blocks):
                start = i * 16
                end = start + 16
                out[x, y, start:end, start:end] = block
    return torch.from_numpy(np.triu(out, 1))


def block_random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.2):
    block = scale * np.random.rand(16, 16)
    block = np.triu(block, k=1)
    out = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                out[x, y, i : i + 16, i : i + 16] = block.copy()
    return torch.from_numpy(out)


def reference_inverse(inp: torch.Tensor) -> torch.Tensor:
    n = inp.shape[-1]
    identity = np.eye(n, dtype=np.double)
    golden = np.zeros(inp.shape, dtype=np.double)
    inp_cpu = inp.cpu()
    for x in range(inp.shape[0]):
        for y in range(inp.shape[1]):
            golden[x, y] = np.linalg.inv(inp_cpu[x, y].numpy().astype(np.double) + identity)
    return torch.from_numpy(golden)


def _run_kernel_flattened(lib, inp_2d_tiles):
    inp_fp16 = inp_2d_tiles.to(torch.float16).contiguous()
    num_tiles = int(inp_fp16.shape[0])
    matrix_size = int(inp_fp16.shape[-1])

    out = torch.zeros_like(inp_fp16, dtype=torch.float32, device=inp_fp16.device)
    minus_identity = torch.zeros(
        (matrix_size, matrix_size), dtype=torch.float16, device=inp_fp16.device
    )
    minus_identity.fill_diagonal_(-1)

    block_dim = max(1, min(num_tiles, 32))
    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        block_dim,
        stream_ptr,
        torch_to_ctypes(out),
        torch_to_ctypes(inp_fp16),
        torch_to_ctypes(minus_identity),
        matrix_size,
        num_tiles,
        0,  # contiguous-tile path
    )
    torch.npu.synchronize()
    return out


def run_kernel(lib, inp_bxby):
    bx, by, n, _ = inp_bxby.shape
    flattened = inp_bxby.reshape(bx * by, n, n)
    out = _run_kernel_flattened(lib, flattened)
    return out.reshape(bx, by, n, n)


def run_kernel_bsnd_fallback(lib, inp_bsnd):
    # The generated DSL kernel keeps the same ABI but currently uses contiguous tile
    # loading; for BSND tests we transform to contiguous [tiles, heads, D, D].
    b, s, n, d = inp_bsnd.shape
    if s % d != 0:
        raise ValueError("S must be divisible by D for BSND fallback testing.")
    tiled = inp_bsnd.reshape(b, s // d, d, n, d).transpose(2, 3).contiguous()
    tiled = tiled.reshape(b * s // d, n, d, d)
    out_tiled = run_kernel(lib, tiled)
    return out_tiled.reshape(b, s // d, n, d, d).transpose(2, 3).reshape(b, s, n, d)


def check_case(lib, U: torch.Tensor, atol: float, rtol: float, ftol: float):
    golden = reference_inverse(U).to(torch.float64)
    actual = run_kernel(lib, U.to(device)).cpu().to(torch.float64)

    frob_error = torch.sqrt(torch.sum((golden - actual) ** 2) / torch.sum(golden * golden))
    assert np.allclose(actual.numpy(), golden.numpy(), atol=atol, rtol=rtol)
    assert frob_error <= ftol, f"frob_error={frob_error}"


def check_case_bsnd(
    lib, U: torch.Tensor, B: int, S: int, N: int, D: int, atol: float, rtol: float, ftol: float
):
    U = U.to(torch.float16)
    golden = reference_inverse(U)

    U_bsnd = U.transpose(1, 2).contiguous().reshape(B, S, N, D)
    golden_bsnd = golden.transpose(1, 2).contiguous().reshape(B, S, N, D)
    actual = run_kernel_bsnd_fallback(lib, U_bsnd.to(device)).cpu().to(torch.float64)
    golden_bsnd = golden_bsnd.to(torch.float64)

    frob_error = torch.sqrt(
        torch.sum((golden_bsnd - actual) ** 2) / torch.sum(golden_bsnd * golden_bsnd)
    )
    assert np.allclose(actual.numpy(), golden_bsnd.numpy(), atol=atol, rtol=rtol)
    assert frob_error <= ftol, f"frob_error={frob_error}"


def run_tests(lib):
    generators = [
        (block_ones_triu_matrix, 0.0, 0.0, 0.0),
        (ones_triu_matrix, 0.0, 0.0, 0.0),
        (block_random_triu_matrix, 5e-5, 0.1, 1e-4),
        (random_triu_matrix, 5e-5, 0.1, 1e-4),
    ]

    for n in (16, 32, 64, 128):
        for bx in (1, 2):
            for by in (2, 4):
                for gen, atol, rtol, ftol in generators:
                    U = gen(n, bx, by)
                    check_case(lib, U, atol, rtol, ftol)
                    print(f"[pass] dense n={n}, bx={bx}, by={by}, gen={gen.__name__}")

    for B in (1,):
        for S in (128, 256):
            for N in (4,):
                for D in (16, 32, 64, 128):
                    if S % D != 0:
                        continue
                    for gen, atol, rtol, ftol in generators:
                        U = gen(D, B * S // D, N)
                        check_case_bsnd(lib, U, B, S, N, D, atol, rtol, ftol)
                        print(f"[pass] bsnd B={B}, S={S}, N={N}, D={D}, gen={gen.__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Use manual-sync library instead of auto-sync library.",
    )
    args = parser.parse_args()

    lib_path = (
        "./rec_unroll_manual_sync_lib.so"
        if args.manual_sync
        else "./rec_unroll_auto_sync_lib.so"
    )
    device = get_test_device()
    torch.npu.set_device(device)

    kernel_lib = load_lib(lib_path)
    run_tests(kernel_lib)
    print(f"All tests passed for {lib_path}.")
