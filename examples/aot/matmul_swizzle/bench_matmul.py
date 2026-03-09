import argparse
import csv
import ctypes
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device


BLOCK_DIM = 24
SWIZZLE_DIRECTION_LIST = [0, 1]
SWIZZLE_COUNT_LIST = [1, 3, 5]
M_LIST = [128 * i for i in range(1, 37, 4)]  # 128, ..., 4224
SHAPES_NK = [
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
]
N_WARMUP = 5
N_REPEAT = 20
DEFAULT_CSV_REL_PATH = Path("outputs") / "csv" / "bench_matmul.csv"
DEFAULT_PLOT_DIR = Path("outputs") / "plots"


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.call_kernel.restype = None

    def matmul_abt(
        a,
        b,
        *,
        block_dim=24,
        swizzle_direction=1,
        swizzle_count=3,
        stream_ptr=None,
    ):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("matmul_abt expects 2D tensors: a[M,K], b[N,K]")
        if a.shape[1] != b.shape[1]:
            raise ValueError(
                f"K mismatch: a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}"
            )
        if a.dtype != torch.float16 or b.dtype != torch.float16:
            raise ValueError("matmul_abt currently supports float16 inputs only")

        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        m = int(a.shape[0])
        k = int(a.shape[1])
        n = int(b.shape[0])
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)

        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            torch_to_ctypes(c),
            m,
            n,
            k,
            swizzle_direction,
            swizzle_count,
        )
        return c

    return matmul_abt


def _parse_int_list(raw: str):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("List cannot be empty.")
    return [int(p) for p in parts]


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AOT matmul_abt vs torch.nn.functional.linear, "
            "save CSV, and optionally plot throughput."
        )
    )
    parser.add_argument(
        "--lib",
        type=str,
        default="matmul_kernel.so",
        help="Path to shared library with call_kernel (default: matmul_kernel.so).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV_REL_PATH),
        help=f"Output CSV path (default: {DEFAULT_CSV_REL_PATH}).",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(DEFAULT_PLOT_DIR),
        help=f"Plot output directory (default: {DEFAULT_PLOT_DIR}).",
    )
    parser.add_argument(
        "--m-list",
        type=str,
        default=",".join(str(m) for m in M_LIST),
        help="Comma-separated M values (default: script M_LIST).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=N_WARMUP,
        help=f"Warmup iterations (default: {N_WARMUP}).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=N_REPEAT,
        help=f"Timed iterations (default: {N_REPEAT}).",
    )
    return parser.parse_args()


def _time_fn(fn, a_list, b_list, warmup, repeat):
    for a, b in zip(a_list[:warmup], b_list[:warmup]):
        fn(a, b)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for a, b in zip(a_list[warmup : warmup + repeat], b_list[warmup : warmup + repeat]):
        fn(a, b)
    end.record()
    torch.npu.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms * 1000.0 / repeat


def _maybe_plot(rows, plot_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    grouped = {}
    for row in rows:
        key = (row["n"], row["k"])
        grouped.setdefault(key, []).append(row)

    for (n, k), chunk in grouped.items():
        m_values = sorted({r["m"] for r in chunk})
        swizzles = sorted(
            {(r["swizzle_direction"], r["swizzle_count"]) for r in chunk},
            key=lambda x: (x[0], x[1]),
        )

        linear_by_m = {}
        for m in m_values:
            candidates = [r for r in chunk if r["m"] == m]
            linear_by_m[m] = sum(r["linear_tflops"] for r in candidates) / len(candidates)

        plt.figure(figsize=(9, 5))
        plt.plot(
            m_values,
            [linear_by_m[m] for m in m_values],
            marker="x",
            linestyle="--",
            color="#111111",
            label="F.linear",
        )

        cmap = plt.get_cmap("tab10")
        for idx, (direction, count) in enumerate(swizzles):
            series = []
            for m in m_values:
                candidates = [
                    r
                    for r in chunk
                    if r["m"] == m
                    and r["swizzle_direction"] == direction
                    and r["swizzle_count"] == count
                ]
                if not candidates:
                    series.append(float("nan"))
                else:
                    series.append(
                        sum(r["custom_tflops"] for r in candidates) / len(candidates)
                    )
            plt.plot(
                m_values,
                series,
                marker="o",
                linestyle="-",
                color=cmap(idx % 10),
                label=f"matmul_abt(d={direction}, c={count})",
            )

        plt.title(f"TFLOPS vs M (N={n}, K={k})")
        plt.xlabel("M")
        plt.ylabel("TFLOPS")
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out = plot_dir / f"flops_n{n}_k{k}.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved plot: {out}")


def main():
    args = _parse_args()
    base_dir = Path(__file__).resolve().parent
    device = get_test_device()
    torch.npu.set_device(device)

    m_list = _parse_int_list(args.m_list)
    if args.warmup < 1 or args.repeat < 1:
        raise ValueError("--warmup and --repeat must be positive integers.")

    lib_path = Path(args.lib)
    if not lib_path.is_absolute():
        lib_path = base_dir / lib_path
    if not lib_path.exists():
        raise FileNotFoundError(f"Kernel library not found: {lib_path}")

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = base_dir / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    plot_dir = Path(args.plot_dir)
    if not plot_dir.is_absolute():
        plot_dir = base_dir / plot_dir

    matmul_abt = load_lib(str(lib_path))
    torch.manual_seed(0)

    rows = []
    total_cases = len(m_list) * len(SHAPES_NK) * len(SWIZZLE_DIRECTION_LIST) * len(
        SWIZZLE_COUNT_LIST
    )
    case_idx = 0

    for n, k in SHAPES_NK:
        for m in m_list:
            alloc = args.warmup + args.repeat
            a_list = [torch.randn(m, k, dtype=torch.float16, device=device) for _ in range(alloc)]
            b_list = [torch.randn(n, k, dtype=torch.float16, device=device) for _ in range(alloc)]
            c_ref = F.linear(a_list[0], b_list[0])
            torch.npu.synchronize()

            linear_time_us = _time_fn(F.linear, a_list, b_list, args.warmup, args.repeat)
            flops = 2.0 * m * n * k
            linear_tflops = flops / linear_time_us / 1e6

            print(f"\n(M,N,K)=({m},{n},{k}) F.linear={linear_tflops:.3f} TFLOPS")

            for swizzle_direction in SWIZZLE_DIRECTION_LIST:
                for swizzle_count in SWIZZLE_COUNT_LIST:
                    case_idx += 1

                    def _custom(a, b, _d=swizzle_direction, _c=swizzle_count):
                        return matmul_abt(
                            a,
                            b,
                            block_dim=BLOCK_DIM,
                            swizzle_direction=_d,
                            swizzle_count=_c,
                        )

                    c = _custom(a_list[0], b_list[0])
                    torch.npu.synchronize()
                    max_absdiff = float((c - c_ref).abs().max().item())
                    mean_absdiff = float((c - c_ref).abs().mean().item())
                    custom_time_us = _time_fn(_custom, a_list, b_list, args.warmup, args.repeat)
                    custom_tflops = flops / custom_time_us / 1e6

                    print(
                        f"  [{case_idx:03d}/{total_cases}] "
                        f"d={swizzle_direction} c={swizzle_count} "
                        f"custom={custom_tflops:.3f} TFLOPS "
                        f"speedup={linear_time_us / custom_time_us:.3f}x "
                        f"mean_diff={mean_absdiff:.3e}"
                    )

                    rows.append(
                        {
                            "m": m,
                            "n": n,
                            "k": k,
                            "block_dim": BLOCK_DIM,
                            "swizzle_direction": swizzle_direction,
                            "swizzle_count": swizzle_count,
                            "linear_time_us": linear_time_us,
                            "linear_tflops": linear_tflops,
                            "custom_time_us": custom_time_us,
                            "custom_tflops": custom_tflops,
                            "speedup_vs_linear": linear_time_us / custom_time_us,
                            "max_absdiff": max_absdiff,
                            "mean_absdiff": mean_absdiff,
                        }
                    )

    fieldnames = [
        "m",
        "n",
        "k",
        "block_dim",
        "swizzle_direction",
        "swizzle_count",
        "linear_time_us",
        "linear_tflops",
        "custom_time_us",
        "custom_tflops",
        "speedup_vs_linear",
        "max_absdiff",
        "mean_absdiff",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved benchmark CSV: {csv_path}")

    _maybe_plot(rows, plot_dir)


if __name__ == "__main__":
    main()
