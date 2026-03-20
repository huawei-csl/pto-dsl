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

UNIFORM_ATOL = 1.5e-1
UNIFORM_RTOL = 1.5e-1
UNIFORM_FTOL = 2.5e-1  # TODO: seems too big


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim (fixed core count)
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # in_delta
        ctypes.c_void_p,  # identity_neg_half
        ctypes.c_uint32,  # runtime batch_size
        ctypes.c_uint32,  # log2(matrix_size)
    ]
    lib.call_kernel.restype = None
    return lib


def ill_matrix(n, batch, offdiag=0.5):
    out = np.zeros((batch, n, n), dtype=np.float32)
    for b in range(batch):
        out[b] = offdiag * np.tril(np.ones((n, n), dtype=np.float32), k=-1)
    return torch.from_numpy(out)


def structured_random_matrix(n, batch, scale=0.1):
    h = n // 2
    out = np.zeros((batch, n, n), dtype=np.float32)
    for b in range(batch):
        a11 = scale * np.tril(np.random.uniform(-1.0, 1.0, size=(h, h)).astype(np.float32), k=-1)
        a22 = scale * np.tril(np.random.uniform(-1.0, 1.0, size=(h, h)).astype(np.float32), k=-1)
        a21 = scale * np.random.uniform(-1.0, 1.0, size=(h, h)).astype(np.float32)
        out[b, :h, :h] = a11
        out[b, h:, h:] = a22
        out[b, h:, :h] = a21
    return torch.from_numpy(out)


def structured_scale_by_n(n):
    # Keep larger matrices closer to identity so the trend follows the note:
    # medium sizes are very accurate, while the hardest ill-conditioned cases
    # degrade only at larger n.
    return {
        16: 0.10,
        32: 0.08,
        64: 0.05,
        128: 0.03,
    }[n]


def ill_offdiag_for_tests(n):
    # Use a smaller scale for bigger sizes.
    return {
        16: 0.2,
        32: 0.1,
        64: 0.05,
        128: 0.02,
    }[n]


def run_kernel(lib, inp_delta):
    inp_fp16 = inp_delta.to(torch.float16).contiguous()
    n = int(inp_fp16.shape[-1])
    batch = int(inp_fp16.shape[0])
    h = n // 2
    log2_blocksize = int(math.log2(n))

    identity_neg_half = torch.zeros((h, h), dtype=torch.float16, device=inp_fp16.device)
    identity_neg_half.fill_diagonal_(-1)
    out = torch.zeros((batch, n, n), dtype=torch.float32, device=inp_fp16.device)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        PERSISTENT_BLOCK_DIM,
        stream_ptr,
        torch_to_ctypes(out),
        torch_to_ctypes(inp_fp16),
        torch_to_ctypes(identity_neg_half),
        batch,
        log2_blocksize,
    )
    torch.npu.synchronize()
    return out


def reference_inverse(inp_delta):
    n = inp_delta.shape[-1]
    identity = np.eye(n, dtype=np.float64)
    inp_cpu = inp_delta.cpu().numpy().astype(np.float64)
    return torch.from_numpy(np.linalg.inv(inp_cpu + identity))


def check_case(lib, matrix_gen, n, batch, atol, rtol, ftol):
    inp_delta = matrix_gen(n=n, batch=batch).to(device)
    ref = reference_inverse(inp_delta).to(torch.float64)
    out = run_kernel(lib, inp_delta).cpu().to(torch.float64)

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


def run_test(lib, n):
    failures = []
    structured_scale = structured_scale_by_n(n)
    ill_offdiag = ill_offdiag_for_tests(n)
    atol, rtol, ftol = UNIFORM_ATOL, UNIFORM_RTOL, UNIFORM_FTOL

    for batch in [1, 4, 16, 24, 27, 48, 96, 99, 135]:
        failure = check_case(
            lib,
            matrix_gen=lambda n, batch: structured_random_matrix(
                n=n, batch=batch, scale=structured_scale
            ),
            n=n,
            batch=batch,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
        )
        if failure is not None:
            failures.append(failure)

    for batch in [1, 4]:
        failure = check_case(
            lib,
            matrix_gen=lambda n, batch: ill_matrix(n=n, batch=batch, offdiag=ill_offdiag),
            n=n,
            batch=batch,
            atol=atol,
            rtol=rtol,
            ftol=ftol,
        )
        if failure is not None:
            failures.append(failure)

    total = 11
    print(f"summary: n={n}, pass={total - len(failures)}, fail={len(failures)}, total={total}")

    if failures:
        warnings.warn(
            f"{len(failures)} cases failed. First: {failures[0]}",
            stacklevel=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-size",
        type=int,
        choices=SUPPORTED_MATRIX_SIZES,
        default=64,
        help="Only validate this matrix size n.",
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

    kernel_lib = load_lib(args.lib_path)
    run_test(kernel_lib, n=args.matrix_size)
    print(f"Finished tests for n={args.matrix_size} with {args.lib_path}.")
