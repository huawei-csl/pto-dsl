"""Numerical-accuracy test for the deepseek_v4 ``act_quant`` PTO kernel."""

import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from act_quant_util import (  # noqa: E402
    BLOCK_SIZE,
    _KERNEL_SO,
    act_quant,
    act_quant_ref,
)


pytestmark = pytest.mark.skipif(
    not _KERNEL_SO.is_file(),
    reason=f"Build kernel first: {_KERNEL_SO}",
)


@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("N", [BLOCK_SIZE, BLOCK_SIZE * 2, BLOCK_SIZE * 4])
def test_act_quant_numerical(npu_device, M, N):
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.float16, device=npu_device)

    y_pto, s_pto = act_quant(x)
    y_ref, s_ref = act_quant_ref(x)

    # Scale: tight tolerance (single fp32 reduction per group).
    torch.testing.assert_close(s_pto, s_ref, rtol=1e-3, atol=1e-6)
    # Quantized values: allow ±1 ULP (rounding direction differs near .5).
    diff = (y_pto.to(torch.int32) - y_ref.to(torch.int32)).abs()
    assert diff.max().item() <= 1, f"max int8 diff = {diff.max().item()}"
    # Most values should match exactly.
    assert (diff == 0).float().mean().item() > 0.95
