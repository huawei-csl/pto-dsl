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
        (1, 1, BLOCK * 2, BLOCK),
        (1, 4, BLOCK * 4, BLOCK * 2),
        (2, 2, BLOCK * 8, BLOCK * 2),
    ]
    for B, M, N, K in cases:
        q = torch.randn(B, M, H_PAD, D, device=device).to(torch.float16)
        kv = torch.randn(B, N, D, device=device).to(torch.float16)
        attn_sink = torch.randn(H_PAD, dtype=torch.float32, device=device)
        topk_idxs = (
            torch.stack([torch.randperm(N, device=device)[:K] for _ in range(B * M)])
            .reshape(B, M, K)
            .to(torch.int32)
        )

        o_pto = sparse_attn(q, kv, attn_sink, topk_idxs, scale)
        o_ref = sparse_attn_ref(q, kv, attn_sink, topk_idxs, scale)
        torch.testing.assert_close(o_pto, o_ref, rtol=5e-3, atol=5e-3)
        print(f"sparse_attn B={B} M={M} N={N} K={K}: OK")
    print("sparse_attn: all shapes PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
