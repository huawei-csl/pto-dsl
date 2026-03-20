import argparse
import ctypes
import math
import random
import warnings

import numpy as np
import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

SUPPORTED_MATRIX_SIZES = (16, 32, 64, 128)
try:
    PERSISTENT_BLOCK_DIM = int(torch.npu.get_device_properties("npu").cube_core_num)
except Exception:
    PERSISTENT_BLOCK_DIM = 24


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim (fixed core count)
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # in_delta (M - I)
        ctypes.c_void_p,  # identity_neg
        ctypes.c_uint32,  # runtime batch_size
        ctypes.c_uint32,  # log2(matrix_size)
    ]
    lib.call_kernel.restype = None
    return lib


def dense_stable_matrix(n, batch, scale=0.02):
    eye = np.eye(n, dtype=np.float32)
    noise = np.random.uniform(-1.0, 1.0, size=(batch, n, n)).astype(np.float32)
    out = eye[None, :, :] + scale * noise
    return torch.from_numpy(out)


def run_kernel(lib, inp):
    inp_fp16 = inp.to(torch.float16).contiguous()
    n = int(inp_fp16.shape[-1])
    batch = int(inp_fp16.shape[0])
    log2_blocksize = int(math.log2(n))

    identity = torch.eye(n, dtype=torch.float16, device=inp_fp16.device)
    in_delta = (inp_fp16 - identity).contiguous()
    identity_neg = (-identity).contiguous()
    out = torch.zeros_like(inp_fp16, dtype=torch.float32, device=inp_fp16.device)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        PERSISTENT_BLOCK_DIM,
        stream_ptr,
        torch_to_ctypes(out),
        torch_to_ctypes(in_delta),
        torch_to_ctypes(identity_neg),
        batch,
        log2_blocksize,
    )
    torch.npu.synchronize()
    return out


def reference_inverse(inp):
    inp_cpu = inp.cpu().numpy().astype(np.float64)
    inv_ref = np.linalg.inv(inp_cpu)
    return torch.from_numpy(inv_ref)


def check_case(lib, n, batch, atol, rtol, ftol):
    inp = dense_stable_matrix(n=n, batch=batch).to(device)
    ref = reference_inverse(inp).to(torch.float64)
    out = run_kernel(lib, inp).cpu().to(torch.float64)

    frob_error = torch.sqrt(torch.sum((ref - out) ** 2) / torch.sum(ref**2))
    allclose_ok = np.allclose(out.numpy(), ref.numpy(), atol=atol, rtol=rtol)
    frob_ok = bool(frob_error <= ftol)

    nan_count = int(torch.isnan(out).sum().item())
    inf_count = int(torch.isinf(out).sum().item())

    if allclose_ok and frob_ok:
        print(f"[pass] n={n}, batch={batch}, frob={float(frob_error):.3e}")
        return None

    msg = (
        f"[fail] n={n}, batch={batch}, frob={float(frob_error):.3e}, "
        f"nan={nan_count}, inf={inf_count}"
    )
    print(msg)
    return msg


def run_test(lib, n, batch_list):
    failures = []
    for batch in batch_list:
        failure = check_case(
            lib,
            n=n,
            batch=batch,
            atol=6e-3,
            rtol=5e-2,
            ftol=8e-3,
        )
        if failure is not None:
            failures.append(failure)

    total = len(batch_list)
    print(f"summary: n={n}, pass={total - len(failures)}, fail={len(failures)}, total={total}")
    if failures:
        warnings.warn(
            f"{len(failures)} cases failed. First: {failures[0]}",
            stacklevel=2,
        )
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-size",
        type=int,
        choices=SUPPORTED_MATRIX_SIZES,
        default=64,
        help="Only validate this dense matrix size n.",
    )
    parser.add_argument(
        "--lib-path",
        type=str,
        default="./inverse_lib.so",
        help="Shared library path produced by compile.sh.",
    )
    args = parser.parse_args()

    device = get_test_device()
    torch.npu.set_device(device)
    batch_list = [1, 8, 24, 27, 48, 96, 99, 135]

    print(f"\n=== validating kernel: {args.lib_path} ===")
    lib = load_lib(args.lib_path)
    failures = run_test(lib, n=args.matrix_size, batch_list=batch_list)
    print(f"\nfinished tests for n={args.matrix_size}, failures={len(failures)}.")
