"""Run the deepseek_v4 ``sparse_attn`` PTO kernel and validate against
the reference. Exits non-zero on mismatch."""

import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from ptodsl.npu_info import get_test_device

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sparse_attn_util import (  # noqa: E402
    BLOCK,
    D,
    H_PAD,
    sparse_attn,
    sparse_attn_ref,
)


def main() -> int:
    device = get_test_device()
    torch.npu.set_device(device)
    torch.manual_seed(0)
    scale = 1.0 / (D**0.5)

    cases = [
        # (B, M, N, K, H, sentinel_frac)
        (1, 1, BLOCK * 2, BLOCK, H_PAD, 0.0),
        (1, 4, BLOCK * 4, BLOCK * 2, H_PAD, 0.0),
        (2, 2, BLOCK * 8, BLOCK * 2, H_PAD, 0.0),
        (8, 2, BLOCK * 4, BLOCK * 2, H_PAD, 0.0),
        # Padded-heads case (TileLang parity: wrapper pads h<16 → 16):
        (4, 2, BLOCK * 4, BLOCK * 2, 8, 0.0),
        (4, 2, BLOCK * 4, BLOCK, 1, 0.0),
        # Sentinel masking: ~25% of top-k slots are -1 (TileLang parity).
        (4, 4, BLOCK * 4, BLOCK * 2, H_PAD, 0.25),
        (8, 2, BLOCK * 4, BLOCK, 4, 0.25),
    ]
    for B, M, N, K, H, sentinel_frac in cases:
        q = torch.randn(B, M, H, D, device=device).to(torch.float16)
        kv = torch.randn(B, N, D, device=device).to(torch.float16)
        attn_sink = torch.randn(H, dtype=torch.float32, device=device)
        topk_idxs = (
            torch.stack([torch.randperm(N, device=device)[:K] for _ in range(B * M)])
            .reshape(B, M, K)
            .to(torch.int32)
        )
        if sentinel_frac > 0.0:
            mask = torch.rand(B, M, K, device=device) < sentinel_frac
            topk_idxs = torch.where(mask, torch.full_like(topk_idxs, -1), topk_idxs)

        o_pto = sparse_attn(q, kv, attn_sink, topk_idxs, scale)
        o_ref = sparse_attn_ref(q, kv, attn_sink, topk_idxs, scale)
        torch.testing.assert_close(o_pto, o_ref, rtol=5e-3, atol=5e-3)
        print(
            f"sparse_attn B={B} M={M} N={N} K={K} H={H} "
            f"sentinel={sentinel_frac}: OK"
        )
    print("sparse_attn: all shapes PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
