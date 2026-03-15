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

MAX_BLOCK_SIZE = 16


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
        ctypes.c_uint32,  # max_block_size
    ]
    lib.call_kernel.restype = None
    return lib


def random_matrix(n, block_dim_x, block_dim_y, scale=0.01):
    return scale * torch.rand((block_dim_x, block_dim_y, n, n))


def block_ones_matrix(n, block_dim_x, block_dim_y):
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


def block_random_matrix(n, block_dim_x, block_dim_y, scale=0.2):
    block = scale * np.random.rand(16, 16)
    block = np.triu(block, k=1)
    out = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                out[x, y, i : i + 16, i : i + 16] = block.copy()
    return torch.from_numpy(out)


def run_kernel(lib, inp):
    inp_half = inp.to(torch.float16).contiguous()
    n = inp_half.shape[-1]
    block_dim = int(inp_half.shape[0] * inp_half.shape[1])

    out = torch.empty_like(inp_half, dtype=torch.float16, device=inp_half.device)
    identity_neg = torch.zeros((n, n), dtype=torch.float16, device=inp_half.device)
    identity_neg.fill_diagonal_(-1)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        block_dim,
        stream_ptr,
        torch_to_ctypes(out),
        torch_to_ctypes(inp_half),
        torch_to_ctypes(identity_neg),
        n,
        MAX_BLOCK_SIZE,
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
            golden[x, y] = np.linalg.inv(inp_cpu[x, y].numpy().astype(np.double) + identity)
    return torch.from_numpy(golden)


def check_case(lib, matrix_gen: Callable, atol: float, rtol: float, ftol: float):
    n_list = [16, 32, 64, 96, 128]
    block_dim_x_list = [1, 3, 7, 16]
    block_dim_y_list = [1, 2, 4, 16]
    for n in n_list:
        for block_dim_x in block_dim_x_list:
            for block_dim_y in block_dim_y_list:
                inp = matrix_gen(n, block_dim_x, block_dim_y).to(device)
                ref = reference_inverse(inp).to(torch.float64)
                out = run_kernel(lib, inp).cpu().to(torch.float64)

                frob_error = torch.sqrt(
                    torch.sum((ref - out) * (ref - out)) / torch.sum(ref * ref)
                )

                np.testing.assert_allclose(
                    out.numpy(),
                    ref.numpy(),
                    atol=atol,
                    rtol=rtol,
                    err_msg=(
                        "allclose mismatch: "
                        f"shape={tuple(inp.shape)}, atol={atol}, rtol={rtol}"
                    ),
                )
                assert frob_error <= ftol, f"frob_error={frob_error}"
                print(
                    f"[pass] n={n}, bx={block_dim_x}, by={block_dim_y}, "
                    f"frob={float(frob_error):.3e}"
                )


def run_test(lib):
    check_case(lib, block_ones_matrix, atol=0.0, rtol=0.0, ftol=0.0)
    check_case(lib, block_random_matrix, atol=5e-5, rtol=0.1, ftol=1e-4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Use manual-sync library instead of the default auto-sync library.",
    )
    args = parser.parse_args()

    lib_path = "./inverse_manual_sync_lib.so" if args.manual_sync else "./inverse_auto_sync_lib.so"
    device = get_test_device()
    torch.npu.set_device(device)

    kernel_lib = load_lib(lib_path)
    run_test(kernel_lib)
    print(f"All tests passed for {lib_path}.")
