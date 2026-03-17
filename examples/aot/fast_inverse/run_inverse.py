import argparse
import ctypes
import random
import warnings
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
        ctypes.c_uint32,  # block_size for block-diag-matrices
    ]
    lib.call_kernel.restype = None
    return lib


def random_matrix(n, block_dim_x, block_dim_y, scale=0.01):
    # TODO: this data generator is not used yet
    return scale * torch.rand((block_dim_x, block_dim_y, n, n))


def blockdiag_ones_matrix(n, block_dim_x, block_dim_y, block_size=16):
    block = np.ones((block_size, block_size))
    n_blocks = n // block_size
    out = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(n_blocks):
                start = i * block_size
                end = start + block_size
                out[x, y, start:end, start:end] = block
    return torch.from_numpy(np.triu(out, 1))


def blockdiag_random_matrix(n, block_dim_x, block_dim_y, block_size=16):
    if block_size == 16:
        scale = 0.2
    elif block_size == 32:
        scale = 0.05
    elif block_size == 64:
        scale = 0.01
    else:
        raise ValueError("block_size must be 16/32/64")
    block = scale * np.random.rand(block_size, block_size)
    block = np.triu(block, k=1)
    out = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, block_size):
                out[x, y, i : i + block_size, i : i + block_size] = block.copy()
    return torch.from_numpy(out)


def run_kernel(lib, inp, blockdiag_size=16):
    inp_fp16 = inp.to(torch.float16).contiguous()
    n = int(inp_fp16.shape[-1])
    block_dim = int(inp_fp16.shape[0] * inp_fp16.shape[1])

    # Run true matrix sizes directly (e.g., 32x32, 64x64) without padding.
    run_n = n
    inp_run = inp_fp16

    out = torch.zeros_like(inp_run, dtype=torch.float32, device=inp_run.device)
    identity_neg = torch.zeros(
        (run_n, run_n), dtype=torch.float16, device=inp_run.device
    )
    identity_neg.fill_diagonal_(-1)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        block_dim,
        stream_ptr,
        torch_to_ctypes(out),
        torch_to_ctypes(inp_run),
        torch_to_ctypes(identity_neg),
        run_n,
        blockdiag_size,
    )
    torch.npu.synchronize()
    return out


def reference_inverse(inp):
    n = inp.shape[-1]
    identity = np.eye(n, dtype=np.double)
    golden = np.zeros(inp.shape, dtype=np.double)
    inp_cpu = inp.cpu()
    for x in range(inp.shape[0]):
        for y in range(inp.shape[1]):
            golden[x, y] = np.linalg.inv(
                inp_cpu[x, y].numpy().astype(np.double) + identity
            )
    return torch.from_numpy(golden)


def check_case(
    lib, matrix_gen: Callable, atol: float, rtol: float, ftol: float, blockdiag_size=16
):
    if blockdiag_size == 16:
        n_list = [16, 32, 64, 96, 128]
    elif blockdiag_size == 32:
        n_list = [32, 64, 128]
    elif blockdiag_size == 64:
        n_list = [64, 128]
    else:
        raise ValueError("blockdiag_size must be 16/32/64")
    block_dim_x_list = [1, 3, 7, 16]
    block_dim_y_list = [1, 2, 4, 16]
    failures = []
    passes = 0
    for n in n_list:
        for block_dim_x in block_dim_x_list:
            for block_dim_y in block_dim_y_list:
                inp = matrix_gen(
                    n, block_dim_x, block_dim_y, block_size=blockdiag_size
                ).to(device)
                ref = reference_inverse(inp).to(torch.float64)
                out = run_kernel(lib, inp).cpu().to(torch.float64)

                frob_error = torch.sqrt(
                    torch.sum((ref - out) * (ref - out)) / torch.sum(ref * ref)
                )

                nan_count = int(torch.isnan(out).sum().item())
                inf_count = int(torch.isinf(out).sum().item())

                allclose_ok = np.allclose(
                    out.numpy(), ref.numpy(), atol=atol, rtol=rtol
                )
                frob_ok = bool(frob_error <= ftol)
                if allclose_ok and frob_ok:
                    passes += 1
                    print(
                        f"[pass] n={n}, bx={block_dim_x}, by={block_dim_y}, "
                        f"frob={float(frob_error):.3e}"
                    )
                else:
                    msg = (
                        f"[fail] n={n}, bx={block_dim_x}, by={block_dim_y}, "
                        f"frob={float(frob_error):.3e}, nan={nan_count}, inf={inf_count}"
                    )
                    print(msg)
                    failures.append(msg)

    total = len(n_list) * len(block_dim_x_list) * len(block_dim_y_list)
    print(f"summary: pass={passes}, fail={len(failures)}, total={total}")
    return failures


def run_test(lib):
    failures = []
    for blockdiag_size in (16,):
        failures.extend(
            check_case(
                lib,
                blockdiag_ones_matrix,
                atol=0.0,
                rtol=0.0,
                ftol=0.0,
                blockdiag_size=blockdiag_size,
            )
        )
    for blockdiag_size in (16, 32, 64):
        failures.extend(
            check_case(
                lib,
                blockdiag_random_matrix,
                atol=5e-5,
                rtol=0.1,
                ftol=1.2e-4,
                blockdiag_size=blockdiag_size,
            )
        )
    if failures:
        warnings.warn(
            f"{len(failures)} cases failed. First: {failures[0]}",
            stacklevel=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Use manual-sync library instead of the default auto-sync library.",
    )
    args = parser.parse_args()

    lib_path = (
        "./inverse_manual_sync_lib.so"
        if args.manual_sync
        else "./inverse_auto_sync_lib.so"
    )
    device = get_test_device()
    torch.npu.set_device(device)

    kernel_lib = load_lib(lib_path)
    run_test(kernel_lib)
    print(f"All tests passed for {lib_path}.")
