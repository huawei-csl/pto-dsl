#!/usr/bin/env python3
"""
Median effective memory bandwidth (GB/s) for Sinkhorn forward:

- **Batched** / **naive**: PTODSL builders from this directory (after ``./compile.sh``).
- **C++ ref**: hand-written ``cpp_ref/kernel_sinkhorn.cpp`` (same ``call_sinkhorn`` ABI).
  Use ``--build-cpp`` to run ``cpp_ref/compile.sh`` if the ``.so`` is missing.

Uses ``jit_util_sinkhorn.bench_sinkhorn_forward_gbs`` (read fp16 + write fp16 per launch).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

_demo_dir = Path(__file__).resolve().parent
if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))

import torch_npu  # noqa: E402, F401

from jit_util_sinkhorn import bench_sinkhorn_forward_gbs, load_sinkhorn_host_lib  # noqa: E402


def cpp_ref_dir() -> Path:
    return _demo_dir / "cpp_ref"


def default_cpp_kernel_so() -> Path:
    return cpp_ref_dir() / "outputs" / "kernel_sinkhorn.so"


def default_cpp_kernel_src() -> Path:
    return cpp_ref_dir() / "kernel_sinkhorn.cpp"


def build_cpp_reference_kernel() -> Path:
    """Run ``cpp_ref/compile.sh`` (bisheng manual-sync batched kernel)."""
    script = cpp_ref_dir() / "compile.sh"
    if not script.is_file():
        raise FileNotFoundError(f"Missing {script}")
    if not default_cpp_kernel_src().is_file():
        raise FileNotFoundError(f"C++ kernel source not found: {default_cpp_kernel_src()}")
    subprocess.run(["bash", str(script)], check=True, cwd=str(cpp_ref_dir()))
    out = default_cpp_kernel_so()
    if not out.is_file():
        raise FileNotFoundError(f"Expected {out} after compile")
    return out


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
    p.add_argument(
        "--cpp-so",
        type=Path,
        default=None,
        help="Path to C++ reference call_sinkhorn .so (default: cpp_ref/outputs/kernel_sinkhorn.so).",
    )
    p.add_argument(
        "--build-cpp",
        action="store_true",
        help="Run cpp_ref/compile.sh when --cpp-so is missing.",
    )
    p.add_argument(
        "--no-cpp",
        action="store_true",
        help="Do not benchmark the hand-written C++ kernel.",
    )
    args = p.parse_args()

    device = _parse_npu(args.npu)
    torch.npu.set_device(device)

    n1_list = [int(x.strip()) for x in args.shapes.split(",") if x.strip()]

    cpp_so: Path | None = None
    cpp_lib = None
    if not args.no_cpp:
        cpp_so = args.cpp_so or default_cpp_kernel_so()
        if not cpp_so.is_file() and args.build_cpp:
            build_cpp_reference_kernel()
            cpp_so = args.cpp_so or default_cpp_kernel_so()
        if cpp_so.is_file():
            cpp_lib = load_sinkhorn_host_lib(cpp_so)
        elif args.cpp_so is not None:
            sys.stderr.write(f"error: C++ .so not found: {cpp_so}\n")
            sys.exit(1)
        else:
            sys.stderr.write(
                f"note: skipping C++ reference (no {cpp_so}). "
                "Run ./compile.sh or: python3 bench_sinkhorn_bandwidth.py --build-cpp\n"
            )

    print(f"device={device} warmup={args.warmup} iters={args.iters} repeat={args.repeat}")
    if cpp_lib is not None:
        print(
            "shape (matrices) | batched GB/s | naive GB/s | C++ ref GB/s | batched/naive | batched/C++"
        )
        print("---|---:|---:|---:|---:|---:")
    else:
        print("shape (matrices) | batched GB/s | naive GB/s | batched/naive")
        print("---|---:|---:|---:")

    for n1 in n1_list:
        torch.manual_seed(1)
        x = torch.randn((1, n1, 4, 4), dtype=torch.float16, device=device)
        bw_b = bench_sinkhorn_forward_gbs(
            x,
            args.repeat,
            args.eps,
            impl="batched",
            warmup=args.warmup,
            iters=args.iters,
        )
        bw_n = bench_sinkhorn_forward_gbs(
            x,
            args.repeat,
            args.eps,
            impl="naive",
            warmup=args.warmup,
            iters=args.iters,
        )
        ratio_bn = bw_b / bw_n if bw_n > 0 else float("nan")
        nmat = n1
        if cpp_lib is not None:
            bw_c = bench_sinkhorn_forward_gbs(
                x,
                args.repeat,
                args.eps,
                lib=cpp_lib,
                warmup=args.warmup,
                iters=args.iters,
            )
            ratio_bc = bw_b / bw_c if bw_c > 0 else float("nan")
            print(
                f"(1, {n1}, 4, 4) ({nmat} matrices) | {bw_b:.3f} | {bw_n:.3f} | {bw_c:.3f} | "
                f"{ratio_bn:.3f} | {ratio_bc:.3f}"
            )
        else:
            print(
                f"(1, {n1}, 4, 4) ({nmat} matrices) | {bw_b:.3f} | {bw_n:.3f} | {ratio_bn:.3f}"
            )


if __name__ == "__main__":
    main()
