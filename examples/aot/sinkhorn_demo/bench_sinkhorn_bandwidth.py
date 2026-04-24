#!/usr/bin/env python3
"""
Median effective memory bandwidth (GB/s) for Sinkhorn forward: batched vs naive.

Uses the same byte accounting and timing style as ``jit_util_sinkhorn.bench_sinkhorn_forward_gbs``
(read fp16 input + write fp16 output per launch). Run after ``./compile.sh``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_demo_dir = Path(__file__).resolve().parent
if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))

import torch_npu  # noqa: E402, F401

from jit_util_sinkhorn import bench_sinkhorn_forward_gbs  # noqa: E402


def _parse_npu(s: str) -> str:
    t = s.strip().strip('"').strip("'")
    if t.lower().startswith("npu:"):
        return f"npu:{int(t.split(':', 1)[1])}"
    return f"npu:{int(t)}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npu",
        default="0",
        help="NPU index or npu:N (default: 0).",
    )
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--repeat", type=int, default=10, help="Sinkhorn repeat count.")
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument(
        "--shapes",
        type=str,
        default="65536,262144",
        help="Comma-separated n1 values for shape (1, n1, 4, 4).",
    )
    args = p.parse_args()

    device = _parse_npu(args.npu)
    torch.npu.set_device(device)

    n1_list = [int(x.strip()) for x in args.shapes.split(",") if x.strip()]

    print(f"device={device} warmup={args.warmup} iters={args.iters} repeat={args.repeat}")
    print("shape (matrices) | batched GB/s | naive GB/s | ratio")
    print("---|---:|---:|---:")

    for n1 in n1_list:
        torch.manual_seed(1)
        x = torch.randn((1, n1, 4, 4), dtype=torch.float16, device=device)
        bw_b = bench_sinkhorn_forward_gbs(
            x, args.repeat, args.eps, impl="batched", warmup=args.warmup, iters=args.iters
        )
        bw_n = bench_sinkhorn_forward_gbs(
            x, args.repeat, args.eps, impl="naive", warmup=args.warmup, iters=args.iters
        )
        ratio = bw_b / bw_n if bw_n > 0 else float("nan")
        nmat = n1
        print(f"(1, {n1}, 4, 4) ({nmat} matrices) | {bw_b:.3f} | {bw_n:.3f} | {ratio:.3f}")


if __name__ == "__main__":
    main()
