"""Numerical-accuracy test for the deepseek_v4 ``sparse_attn`` PTO kernel.

The NPU kernel implements full FlashAttention with indexed (top-k) KV
gather: per-query streaming online softmax over the K positions, with
the per-head ``attn_sink`` logit folded into the softmax denominator
but dropped from the V mix (matches the GPU reference).
"""

import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sparse_attn_util import (  # noqa: E402
    BLOCK,
    D,
    H_PAD,
    _KERNEL_SO,
    sparse_attn,
    sparse_attn_ref,
)


pytestmark = pytest.mark.skipif(
    not _KERNEL_SO.is_file(),
    reason=f"Build kernel first: {_KERNEL_SO}",
)


@pytest.mark.parametrize(
    "B,M,N,K",
    [
        (1, 1, BLOCK * 2, BLOCK),
        (1, 4, BLOCK * 4, BLOCK * 2),
        (2, 2, BLOCK * 8, BLOCK * 2),
    ],
)
def test_sparse_attn_numerical(npu_device, B, M, N, K):
    torch.manual_seed(0)
    scale = 1.0 / (D**0.5)

    # Larger magnitudes so a zero-output kernel cannot pass numerically.
    q = torch.randn(B, M, H_PAD, D, device=npu_device).to(torch.float16)
    kv = torch.randn(B, N, D, device=npu_device).to(torch.float16)
    attn_sink = torch.randn(H_PAD, dtype=torch.float32, device=npu_device)
    topk_idxs = (
        torch.stack([torch.randperm(N, device=npu_device)[:K] for _ in range(B * M)])
        .reshape(B, M, K)
        .to(torch.int32)
    )

    o_pto = sparse_attn(q, kv, attn_sink, topk_idxs, scale)
    o_ref = sparse_attn_ref(q, kv, attn_sink, topk_idxs, scale)

    # Tight tolerance so a kernel that writes zeros cannot accidentally
    # pass (reference output is O(1)).
    torch.testing.assert_close(o_pto, o_ref, rtol=5e-3, atol=5e-3)
