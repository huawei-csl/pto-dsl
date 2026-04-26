"""Numerical-accuracy test for the deepseek_v4 ``fp8_gemm`` PTO kernel
(host-side scale-fusion design — see fp8_gemm_util.py)."""

import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from fp8_gemm_util import (  # noqa: E402
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    _KERNEL_SO,
    fp8_gemm,
    fp8_gemm_ref,
)


pytestmark = pytest.mark.skipif(
    not _KERNEL_SO.is_file(),
    reason=f"Build kernel first: {_KERNEL_SO}",
)


@pytest.mark.parametrize(
    "M,N,K",
    [
        (BLOCK_M, BLOCK_N, BLOCK_K),
        (BLOCK_M * 2, BLOCK_N * 2, BLOCK_K * 2),
        (BLOCK_M * 4, BLOCK_N, BLOCK_K * 4),
        (BLOCK_M, BLOCK_N * 4, BLOCK_K),
    ],
)
def test_fp8_gemm_with_scales(npu_device, M, N, K):
    """Non-trivial Sa/Sb to verify scale-fusion semantics end-to-end."""
    torch.manual_seed(0)
    a = (torch.randn(M, K, device=npu_device) * 0.1).to(torch.float16)
    b = (torch.randn(K, N, device=npu_device) * 0.1).to(torch.float16)
    # Realistic per-block scales: positive, ~lognormal centered at 1.
    sa = torch.randn(M, K // BLOCK_K, device=npu_device).exp().to(torch.float32)
    sb = (
        torch.randn(K // BLOCK_K, N // BLOCK_N, device=npu_device)
        .exp()
        .to(torch.float32)
    )

    c_pto = fp8_gemm(a, b, sa, sb)
    c_ref = fp8_gemm_ref(a, b, sa, sb)

    # Tolerance accounts for fp16 round-trip after pre-scaling.
    torch.testing.assert_close(c_pto, c_ref, rtol=2e-2, atol=2e-2)


def test_fp8_gemm_unit_scales(npu_device):
    """Sa = Sb = 1 -> exact bare GEMM behavior."""
    torch.manual_seed(1)
    M, N, K = BLOCK_M * 2, BLOCK_N, BLOCK_K * 2
    a = (torch.randn(M, K, device=npu_device) * 0.1).to(torch.float16)
    b = (torch.randn(K, N, device=npu_device) * 0.1).to(torch.float16)
    sa = torch.ones(M, K // BLOCK_K, device=npu_device, dtype=torch.float32)
    sb = torch.ones(K // BLOCK_K, N // BLOCK_N, device=npu_device, dtype=torch.float32)

    c_pto = fp8_gemm(a, b, sa, sb)
    c_ref = fp8_gemm_ref(a, b, sa, sb)
    torch.testing.assert_close(c_pto, c_ref, rtol=2e-2, atol=2e-2)
