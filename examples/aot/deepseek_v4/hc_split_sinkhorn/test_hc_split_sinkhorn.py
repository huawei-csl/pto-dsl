"""Numerical-accuracy test for the deepseek_v4 ``hc_split_sinkhorn`` kernel.

The PTO kernel runs the sinkhorn iteration on the device; the two sigmoid
heads and the affine fill are computed on the host. ``hc_split_sinkhorn``
in the util stitches them together so the test compares the full
``(pre, post, comb)`` triple against the GPU TileLang semantics.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hc_split_sinkhorn_util import (  # noqa: E402
    HC,
    MIX_HC,
    _KERNEL_SO,
    hc_split_sinkhorn,
    hc_split_sinkhorn_ref,
)


pytestmark = pytest.mark.skipif(
    not _KERNEL_SO.is_file(),
    reason=f"Build kernel first: {_KERNEL_SO}",
)


@pytest.mark.parametrize("n", [16, 64, 256, 1024])
def test_hc_split_sinkhorn_numerical(npu_device, n):
    torch.manual_seed(0)
    mixes = torch.randn(n, MIX_HC, dtype=torch.float32, device=npu_device)
    hc_scale = torch.randn(3, dtype=torch.float32, device=npu_device) * 0.5
    hc_base = torch.randn(MIX_HC, dtype=torch.float32, device=npu_device) * 0.1

    pre_pto, post_pto, comb_pto = hc_split_sinkhorn(mixes, hc_scale, hc_base)
    pre_ref, post_ref, comb_ref = hc_split_sinkhorn_ref(mixes, hc_scale, hc_base)

    torch.testing.assert_close(pre_pto, pre_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(post_pto, post_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(comb_pto, comb_ref, rtol=1e-3, atol=1e-5)
