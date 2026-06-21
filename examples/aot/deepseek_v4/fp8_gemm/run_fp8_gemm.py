"""Run the deepseek_v4 ``fp8_gemm`` PTO kernel and validate against the
reference. Exits non-zero on mismatch."""

import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from ptodsl.npu_info import get_test_device

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fp8_gemm_util import (  # noqa: E402
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    fp8_gemm,
    fp8_gemm_ref,
)


def _check(M, N, K, sa_unit: bool, device, seed: int):
    torch.manual_seed(seed)
    a = (torch.randn(M, K, device=device) * 0.1).to(torch.float16)
    b = (torch.randn(K, N, device=device) * 0.1).to(torch.float16)
    if sa_unit:
        sa = torch.ones(M, K // BLOCK_K, device=device, dtype=torch.float32)
        sb = torch.ones(K // BLOCK_K, N // BLOCK_N, device=device, dtype=torch.float32)
        tag = "unit-scales"
    else:
        sa = torch.randn(M, K // BLOCK_K, device=device).exp().to(torch.float32)
        sb = (
            torch.randn(K // BLOCK_K, N // BLOCK_N, device=device)
            .exp()
            .to(torch.float32)
        )
        tag = "rand-scales"
    c_pto = fp8_gemm(a, b, sa, sb)
    c_ref = fp8_gemm_ref(a, b, sa, sb)
    torch.testing.assert_close(c_pto, c_ref, rtol=2e-2, atol=2e-2)
    print(f"fp8_gemm M={M} N={N} K={K} {tag}: OK")


def main() -> int:
    device = get_test_device()
    torch.npu.set_device(device)

    cases = [
        (BLOCK_M, BLOCK_N, BLOCK_K),
        (BLOCK_M * 2, BLOCK_N * 2, BLOCK_K * 2),
        (BLOCK_M * 4, BLOCK_N, BLOCK_K * 4),
        (BLOCK_M, BLOCK_N * 4, BLOCK_K),
    ]
    for i, (M, N, K) in enumerate(cases):
        _check(M, N, K, sa_unit=False, device=device, seed=i)

    _check(BLOCK_M * 2, BLOCK_N, BLOCK_K * 2, sa_unit=True, device=device, seed=42)
    print("fp8_gemm: all shapes PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
