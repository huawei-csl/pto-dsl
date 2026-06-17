#!/usr/bin/env python3
"""Median effective memory bandwidth (GB/s) for TileLang Sinkhorn forward (float32).

Counts one read and one write of the flattened ``(n1, hc, hc)`` tensor per timed
forward (``2 * n1 * hc * hc * sizeof(float)`` bytes), using the median wall-clock
latency across ``--iters`` runs after ``--warmup``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))

import torch_npu  # noqa: E402, F401

import tilelang  # noqa: E402
from sinkhorn_kernel import build_sinkhorn, sinkhorn_normalize_tilelang  # noqa: E402


def bandwidth_gbs(data_bytes: int, duration_us: float) -> float:
    return (data_bytes / 1e9) / (duration_us / 1e6) if duration_us > 0 else 0.0


def bench_tilelang_sinkhorn_gbs(
    x: torch.Tensor,
    repeat: int,
    eps: float,
    *,
    hc: int,
    warmup: int = 8,
    iters: int = 24,
) -> float:
    """Median GB/s (decimal 1e9) for one forward (chunked launches if needed)."""
    fn = build_sinkhorn(hc=hc, sinkhorn_iters=repeat, eps=eps)
    x_flat = x.reshape(-1, hc, hc).contiguous()
    out_flat = torch.empty_like(x_flat)
    nmat = x_flat.shape[0]
    elem_sz = x_flat.element_size()
    bytes_rw = 2 * nmat * hc * hc * elem_sz

    for _ in range(warmup):
        sinkhorn_normalize_tilelang(
            x_flat, hc=hc, sinkhorn_iters=repeat, eps=eps, _fn=fn, out=out_flat
        )
    torch.npu.synchronize()

    times_us: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sinkhorn_normalize_tilelang(
            x_flat, hc=hc, sinkhorn_iters=repeat, eps=eps, _fn=fn, out=out_flat
        )
        torch.npu.synchronize()
        times_us.append((time.perf_counter() - t0) * 1e6)

    times_us.sort()
    med_us = times_us[len(times_us) // 2]
    return bandwidth_gbs(bytes_rw, med_us)


def _parse_npu(s: str) -> str:
    t = s.strip().strip('"').strip("'")
    if t.lower().startswith("npu:"):
        return f"npu:{int(t.split(':', 1)[1])}"
    return f"npu:{int(t)}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npu", default="0", help="NPU index or npu:N (default: 0).")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--repeat", type=int, default=10, help="Sinkhorn iteration count.")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--hc", type=int, default=4, help="Inner Sinkhorn size (hc x hc).")
    p.add_argument(
        "--shapes",
        type=str,
        default="65536,262144",
        help="Comma-separated n1 values for shape (1, n1, hc, hc).",
    )
    args = p.parse_args()

    device = _parse_npu(args.npu)
    torch.npu.set_device(device)
    tilelang.disable_cache()

    n1_list = [int(x.strip()) for x in args.shapes.split(",") if x.strip()]
    hc = args.hc

    print(
        f"device={device} warmup={args.warmup} iters={args.iters} "
        f"repeat={args.repeat} hc={hc} dtype=float32"
    )
    print("shape (matrices) | TileLang GB/s")
    print("---|---:")

    for n1 in n1_list:
        torch.manual_seed(1)
        x = torch.randn((1, n1, hc, hc), dtype=torch.float32, device=device)
        bw = bench_tilelang_sinkhorn_gbs(
            x,
            args.repeat,
            args.eps,
            hc=hc,
            warmup=args.warmup,
            iters=args.iters,
        )
        print(f"(1, {n1}, {hc}, {hc}) ({n1} matrices) | {bw:.3f}")


if __name__ == "__main__":
    main()
