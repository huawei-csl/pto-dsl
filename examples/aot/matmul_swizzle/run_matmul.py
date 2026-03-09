import ctypes
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_npu

from ptodsl.test_util import get_test_device


BLOCK_DIM_LIST = [1, 20, 24]
SWIZZLE_COUNT_LIST = [1, 3, 5]
M_LIST = [128 * i for i in range(1, 37, 4)]  # 128, ..., 4224
SHAPES_NK = [
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
]
MAX_ABSDIFF_THRESHOLD = 0.5
MEAN_ABSDIFF_THRESHOLD = 1e-4


@dataclass
class CaseResult:
    m: int
    n: int
    k: int
    block_dim: int
    swizzle_count: int
    max_absdiff: float
    mean_absdiff: float


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
        ctypes.c_int,
    ]
    lib.call_kernel.restype = None

    def matmul_abt(a, b, *, block_dim=24, swizzle_count=3, stream_ptr=None):
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
            swizzle_count,
        )
        return c

    return matmul_abt


def run_case(matmul_abt, a, b, c_ref, *, block_dim, swizzle_count):
    c = matmul_abt(a, b, block_dim=block_dim, swizzle_count=swizzle_count)
    torch.npu.synchronize()
    return CaseResult(
        m=int(a.shape[0]),
        n=int(b.shape[0]),
        k=int(a.shape[1]),
        block_dim=block_dim,
        swizzle_count=swizzle_count,
        max_absdiff=float((c - c_ref).abs().max().item()),
        mean_absdiff=float((c - c_ref).abs().mean().item()),
    )


def test_matmul():
    device = get_test_device()
    torch.npu.set_device(device)
    matmul_abt = load_lib("./matmul_kernel.so")

    torch.manual_seed(0)
    checked_cases = 0
    global_worst = None

    for m in M_LIST:
        for n, k in SHAPES_NK:
            a = torch.randn(m, k, dtype=torch.float16, device=device)
            b = torch.randn(n, k, dtype=torch.float16, device=device)
            c_ref = F.linear(a, b)
            torch.npu.synchronize()

            shape_worst = None
            for block_dim in BLOCK_DIM_LIST:
                for swizzle_count in SWIZZLE_COUNT_LIST:
                    result = run_case(
                        matmul_abt,
                        a,
                        b,
                        c_ref,
                        block_dim=block_dim,
                        swizzle_count=swizzle_count,
                    )
                    checked_cases += 1

                    if (
                        shape_worst is None
                        or result.max_absdiff > shape_worst.max_absdiff
                        or (
                            result.max_absdiff == shape_worst.max_absdiff
                            and result.mean_absdiff > shape_worst.mean_absdiff
                        )
                    ):
                        shape_worst = result

                    if (
                        global_worst is None
                        or result.max_absdiff > global_worst.max_absdiff
                        or (
                            result.max_absdiff == global_worst.max_absdiff
                            and result.mean_absdiff > global_worst.mean_absdiff
                        )
                    ):
                        global_worst = result

            print(
                f"(m, n, k)=({m}, {n}, {k}) "
                f"worst(block_dim, swizzle_count)=({shape_worst.block_dim}, "
                f"{shape_worst.swizzle_count}) "
                f"max_absdiff={shape_worst.max_absdiff:.6f} "
                f"mean_absdiff={shape_worst.mean_absdiff:.6f}"
            )

    print(f"checked_cases={checked_cases}")
    print(
        "global_worst "
        f"max_absdiff={global_worst.max_absdiff:.6f} "
        f"mean_absdiff={global_worst.mean_absdiff:.6f} "
        f"at (m, n, k, block_dim, swizzle_count)=({global_worst.m}, "
        f"{global_worst.n}, {global_worst.k}, {global_worst.block_dim}, "
        f"{global_worst.swizzle_count})"
    )

    if global_worst.max_absdiff > MAX_ABSDIFF_THRESHOLD:
        raise AssertionError(
            f"max_absdiff {global_worst.max_absdiff:.6f} exceeds "
            f"threshold {MAX_ABSDIFF_THRESHOLD:.6f}"
        )
    if global_worst.mean_absdiff > MEAN_ABSDIFF_THRESHOLD:
        raise AssertionError(
            f"mean_absdiff {global_worst.mean_absdiff:.6f} exceeds "
            f"threshold {MEAN_ABSDIFF_THRESHOLD:.6f}"
        )


if __name__ == "__main__":
    test_matmul()
