"""Microbenchmark for the deepseek_v4 ``sparse_attn`` PTO kernel.

Three baselines are timed on every shape:

* ``pto``     — the on-device PTO kernel from ``sparse_attn_util``.
* ``ref``     — the eager PyTorch reference from ``sparse_attn_util``
                (small-matmul softmax, slow but exact).
* ``fused``   — the realistic NPU-PyTorch implementation a user would
                actually write: ``torch.gather`` of the K KV rows
                followed by ``torch_npu.npu_fused_infer_attention_score``
                with ``num_key_value_heads=1`` (MQA). The fused op does
                not expose a per-head additive sink logit, so this
                baseline drops the sink term \u2014 it is included only as
                a *speed* baseline and is not a numerical reference.

Run::

    cd examples/aot/deepseek_v4/sparse_attn
    bash compile.sh
    python bench_sparse_attn.py
"""

import sys
from pathlib import Path

import torch
import torch_npu

from ptodsl import do_bench
from ptodsl.utils.npu_info import get_test_device

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from sparse_attn_util import (  # noqa: E402
    D,
    H_PAD,
    _KERNEL_SO,
    sparse_attn,
    sparse_attn_ref,
)


def fused_sparse_attn(q, kv, idx, scale):
    """Realistic PyTorch-on-NPU baseline: gather + fused attention.

    Drops the per-head sink logit (not expressible in
    ``npu_fused_infer_attention_score``).

    Args:
        q:   [B, M, H_PAD, D] fp16
        kv:  [B, N, D]        fp16    (single KV head)
        idx: [B, M, K]        int32   (positions into N)
        scale: float

    Returns:
        out: [B, M, H_PAD, D] fp16
    """
    B, M, H, Dq = q.shape
    K = idx.shape[-1]
    # Gather kv[b, idx[b, m]] \u2192 [B, M, K, D].
    idx_long = idx.to(torch.long)
    bidx = torch.arange(B, device=q.device).view(B, 1, 1).expand(B, M, K)
    kv_sel = kv[bidx, idx_long]  # [B, M, K, D]
    # Flatten (B, M) into one batch axis for the fused op.
    bm = B * M
    q_bsh = q.reshape(bm, 1, H * Dq).contiguous()  # BSH, S=1
    k_bsh = kv_sel.reshape(bm, K, Dq).contiguous()  # BSH, kv_heads=1
    v_bsh = k_bsh
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q_bsh,
        k_bsh,
        v_bsh,
        num_heads=H,
        num_key_value_heads=1,
        input_layout="BSH",
        scale=scale,
    )
    return out.reshape(B, M, H, Dq)


SHAPES = [
    # (B, M, N, K)
    (1, 1, 128, 64),
    (1, 4, 256, 128),
    (2, 2, 512, 128),
    (4, 4, 1024, 128),
    (8, 8, 2048, 128),
]


def _alloc(B, M, N, K, device):
    torch.manual_seed(0)
    q = torch.randn(B, M, H_PAD, D, device=device).to(torch.float16)
    kv = torch.randn(B, N, D, device=device).to(torch.float16)
    sink = torch.randn(H_PAD, dtype=torch.float32, device=device)
    idx = torch.randint(0, N, (B, M, K), dtype=torch.int32, device=device)
    scale = 1.0 / (D**0.5)
    return q, kv, sink, idx, scale


def main():
    if not _KERNEL_SO.is_file():
        raise SystemExit(f"Build kernel first: cd {_HERE} && bash compile.sh")
    device = get_test_device()
    torch.npu.set_device(device)

    print(
        f"{'B':>3} {'M':>3} {'N':>5} {'K':>4}"
        f" {'pto us':>10} {'ref us':>10} {'fused us':>10}"
        f" {'pto/ref':>9} {'pto/fused':>10}"
    )
    print("-" * 72)
    for B, M, N, K in SHAPES:
        q, kv, sink, idx, scale = _alloc(B, M, N, K, device)
        pto_us = do_bench(
            lambda: sparse_attn(q, kv, sink, idx, scale),
            warmup_iters=5,
            benchmark_iters=50,
            unit="us",
        )
        ref_us = do_bench(
            lambda: sparse_attn_ref(q, kv, sink, idx, scale),
            warmup_iters=5,
            benchmark_iters=50,
            unit="us",
        )
        try:
            fused_us = do_bench(
                lambda: fused_sparse_attn(q, kv, idx, scale),
                warmup_iters=5,
                benchmark_iters=50,
                unit="us",
            )
            fused_str = f"{fused_us:>10.2f}"
            fused_sp = f"{fused_us / pto_us:>9.2f}x"
        except Exception as e:  # noqa: BLE001
            fused_str = f"{'fail':>10}"
            fused_sp = f"{'-':>10}"
            if B == SHAPES[0][0] and M == SHAPES[0][1]:
                print(f"  (fused baseline failed: {e})")
        print(
            f"{B:>3} {M:>3} {N:>5} {K:>4}"
            f" {pto_us:>10.2f} {ref_us:>10.2f} {fused_str}"
            f" {ref_us / pto_us:>8.2f}x {fused_sp}"
        )


if __name__ == "__main__":
    main()
