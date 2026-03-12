import argparse
import ctypes
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device


BLOCK_DIM = 24
M_LIST = [128 * i for i in range(1, 37, 4)]  # 128, ..., 4224
SHAPES_NK = [
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
]
N_WARMUP = 5
N_REPEAT = 20
PLOT_SHAPES_NK = [(8192, 8192), (16384, 16384)]
DEFAULT_PLOT_DIR = Path("fig")


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
    ]
    lib.call_kernel.restype = None

    def matmul_abt(a, b, *, block_dim=BLOCK_DIM, stream_ptr=None):
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
        )
        return c

    return matmul_abt


def _parse_int_list(raw):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("List cannot be empty.")
    return [int(p) for p in parts]


def _time_us(fn, a_list, b_list, warmup, repeat):
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
    return start.elapsed_time(end) * 1000.0 / repeat


def _maybe_plot(rows, plot_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    style_candidates = ("seaborn-v0_8-whitegrid", "seaborn-whitegrid")
    for style_name in style_candidates:
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue

    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plot_dir.mkdir(parents=True, exist_ok=True)
    title_scale = 1.5
    axis_label_scale = 1.5
    legend_scale = 2.0

    step_defs = [
        ("step1", "single_auto_noswizzle_tflops", "Step1 Kernel", "flops_step1_baseline.png"),
        ("step2", "double_auto_noswizzle_tflops", "Step2 Kernel", "flops_step2_doublebuf.png"),
        ("step3", "double_auto_swizzle_tflops", "Step3 Kernel", "flops_step3_swizzle.png"),
        ("step4", "double_manual_swizzle_tflops", "Step4 Kernel", "flops_step4_manual_pipeline.png"),
    ]

    for _, custom_key, custom_label, out_name in step_defs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, (n, k) in zip(axes, PLOT_SHAPES_NK):
            base_title_size = ax.title.get_size()
            base_label_size = ax.xaxis.label.get_size()
            chunk = [r for r in rows if r["n"] == n and r["k"] == k]
            if not chunk:
                ax.set_title(f"TFLOPS vs M (N={n}, K={k})", fontsize=base_title_size * title_scale)
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                ax.set_xlabel("M", fontsize=base_label_size * axis_label_scale)
                ax.set_ylabel("TFLOPS", fontsize=base_label_size * axis_label_scale)
                ax.grid(alpha=0.25)
                continue

            chunk = sorted(chunk, key=lambda r: r["m"])
            m_values = [r["m"] for r in chunk]
            matmul_tflops = [r["torch_matmul_tflops"] for r in chunk]
            custom_tflops = [r[custom_key] for r in chunk]

            ax.plot(
                m_values,
                matmul_tflops,
                marker="x",
                linestyle="--",
                color="#111111",
                label="torch.matmul",
            )
            ax.plot(
                m_values,
                custom_tflops,
                marker="o",
                linestyle="-",
                color="#1f77b4",
                label=custom_label,
            )
            ax.set_title(f"TFLOPS vs M (N={n}, K={k})", fontsize=base_title_size * title_scale)
            ax.set_xlabel("M", fontsize=base_label_size * axis_label_scale)
            ax.set_ylabel("TFLOPS", fontsize=base_label_size * axis_label_scale)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8 * legend_scale)

        plt.tight_layout()
        out = plot_dir / out_name
        plt.savefig(out, dpi=160, format="png")
        plt.close(fig)
        print(f"Saved plot: {out}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Stepwise performance benchmark for buffering, swizzle, and manual sync."
    )
    parser.add_argument(
        "--double-auto-swizzle-lib",
        type=str,
        default="./build_artifacts/step3_swizzle_kernel.so",
        help="Path to double-buffer auto-sync swizzled shared library.",
    )
    parser.add_argument(
        "--double-auto-noswizzle-lib",
        type=str,
        default="./build_artifacts/step2_doublebuffer_kernel.so",
        help="Path to double-buffer auto-sync non-swizzle shared library.",
    )
    parser.add_argument(
        "--double-manual-swizzle-lib",
        type=str,
        default="./build_artifacts/step4_manual_pipelining_kernel.so",
        help="Path to double-buffer manual-sync swizzled shared library.",
    )
    parser.add_argument(
        "--single-auto-noswizzle-lib",
        type=str,
        default="./build_artifacts/step1_baseline_kernel.so",
        help="Path to single-buffer auto-sync non-swizzle shared library.",
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
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(DEFAULT_PLOT_DIR),
        help=f"Plot output directory (default: {DEFAULT_PLOT_DIR}).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.warmup < 1 or args.repeat < 1:
        raise ValueError("--warmup and --repeat must be positive integers.")

    base_dir = Path(__file__).resolve().parent

    double_auto_swizzle_lib = Path(args.double_auto_swizzle_lib)
    if not double_auto_swizzle_lib.is_absolute():
        double_auto_swizzle_lib = base_dir / double_auto_swizzle_lib
    double_auto_noswizzle_lib = Path(args.double_auto_noswizzle_lib)
    if not double_auto_noswizzle_lib.is_absolute():
        double_auto_noswizzle_lib = base_dir / double_auto_noswizzle_lib
    double_manual_swizzle_lib = Path(args.double_manual_swizzle_lib)
    if not double_manual_swizzle_lib.is_absolute():
        double_manual_swizzle_lib = base_dir / double_manual_swizzle_lib
    single_auto_noswizzle_lib = Path(args.single_auto_noswizzle_lib)
    if not single_auto_noswizzle_lib.is_absolute():
        single_auto_noswizzle_lib = base_dir / single_auto_noswizzle_lib
    if not double_auto_swizzle_lib.exists():
        raise FileNotFoundError(f"Double-buffer auto-sync swizzle library not found: {double_auto_swizzle_lib}")
    if not double_auto_noswizzle_lib.exists():
        raise FileNotFoundError(
            f"Double-buffer auto-sync non-swizzle library not found: {double_auto_noswizzle_lib}"
        )
    if not double_manual_swizzle_lib.exists():
        raise FileNotFoundError(
            f"Double-buffer manual-sync swizzle library not found: {double_manual_swizzle_lib}"
        )
    if not single_auto_noswizzle_lib.exists():
        raise FileNotFoundError(
            f"Single-buffer auto-sync non-swizzle library not found: {single_auto_noswizzle_lib}"
        )
    plot_dir = Path(args.plot_dir)
    if not plot_dir.is_absolute():
        plot_dir = base_dir / plot_dir

    device = get_test_device()
    torch.npu.set_device(device)
    torch.manual_seed(0)

    double_auto_swizzle_mm = load_lib(str(double_auto_swizzle_lib))
    double_auto_noswizzle_mm = load_lib(str(double_auto_noswizzle_lib))
    double_manual_swizzle_mm = load_lib(str(double_manual_swizzle_lib))
    single_auto_noswizzle_mm = load_lib(str(single_auto_noswizzle_lib))
    m_list = _parse_int_list(args.m_list)

    ratios_step1_double_vs_single_noswizzle = []
    ratios_step2_swizzle_vs_noswizzle = []
    ratios_step3_manual_vs_auto_swizzle = []
    plot_rows = []
    print(f"double-buffer auto-sync swizzle lib:      {double_auto_swizzle_lib}")
    print(f"double-buffer auto-sync non-swizzle lib:  {double_auto_noswizzle_lib}")
    print(f"double-buffer manual-sync swizzle lib:    {double_manual_swizzle_lib}")
    print(f"single-buffer auto-sync non-swizzle lib:  {single_auto_noswizzle_lib}")
    print("")

    for n, k in SHAPES_NK:
        print(f"=== N={n}, K={k} ===")
        for m in m_list:
            alloc = args.warmup + args.repeat
            a_list = [torch.randn(m, k, dtype=torch.float16, device=device) for _ in range(alloc)]
            b_list = [torch.randn(n, k, dtype=torch.float16, device=device) for _ in range(alloc)]

            double_auto_swizzle_us = _time_us(
                double_auto_swizzle_mm, a_list, b_list, args.warmup, args.repeat
            )
            double_auto_noswizzle_us = _time_us(
                double_auto_noswizzle_mm, a_list, b_list, args.warmup, args.repeat
            )
            double_manual_swizzle_us = _time_us(
                double_manual_swizzle_mm, a_list, b_list, args.warmup, args.repeat
            )
            single_auto_noswizzle_us = _time_us(
                single_auto_noswizzle_mm, a_list, b_list, args.warmup, args.repeat
            )
            torch_matmul_us = _time_us(
                lambda a, b: torch.matmul(a, b.transpose(0, 1)),
                a_list,
                b_list,
                args.warmup,
                args.repeat,
            )

            flops = 2.0 * m * n * k
            double_auto_swizzle_tflops = flops / double_auto_swizzle_us / 1e6
            double_auto_noswizzle_tflops = flops / double_auto_noswizzle_us / 1e6
            double_manual_swizzle_tflops = flops / double_manual_swizzle_us / 1e6
            single_auto_noswizzle_tflops = flops / single_auto_noswizzle_us / 1e6
            torch_matmul_tflops = flops / torch_matmul_us / 1e6

            # Step 1: buffering effect (double-buffer vs single-buffer, both non-swizzle auto-sync).
            step1_double_vs_single = double_auto_noswizzle_tflops / single_auto_noswizzle_tflops
            # Step 2: swizzle effect (double-buffer auto-sync swizzle vs non-swizzle).
            step2_swizzle_vs_noswizzle = double_auto_swizzle_tflops / double_auto_noswizzle_tflops
            # Step 3: manual-sync effect (double-buffer swizzle manual-sync vs auto-sync).
            step3_manual_vs_auto = double_manual_swizzle_tflops / double_auto_swizzle_tflops

            ratios_step1_double_vs_single_noswizzle.append(step1_double_vs_single)
            ratios_step2_swizzle_vs_noswizzle.append(step2_swizzle_vs_noswizzle)
            ratios_step3_manual_vs_auto_swizzle.append(step3_manual_vs_auto)
            plot_rows.append(
                {
                    "m": m,
                    "n": n,
                    "k": k,
                    "torch_matmul_tflops": torch_matmul_tflops,
                    "single_auto_noswizzle_tflops": single_auto_noswizzle_tflops,
                    "double_auto_noswizzle_tflops": double_auto_noswizzle_tflops,
                    "double_auto_swizzle_tflops": double_auto_swizzle_tflops,
                    "double_manual_swizzle_tflops": double_manual_swizzle_tflops,
                }
            )

            print(
                f"(M,N,K)=({m},{n},{k}) "
                f"single_noswizzle={single_auto_noswizzle_tflops:.3f}TF, "
                f"double_noswizzle_auto={double_auto_noswizzle_tflops:.3f}TF, "
                f"double_swizzle_auto={double_auto_swizzle_tflops:.3f}TF, "
                f"double_swizzle_manual={double_manual_swizzle_tflops:.3f}TF, "
                f"torch_matmul={torch_matmul_tflops:.3f}TF, "
                f"step1_ratio(double_noswizzle_auto/single_noswizzle)={step1_double_vs_single:.3f}x, "
                f"step2_ratio(double_swizzle_auto/double_noswizzle_auto)={step2_swizzle_vs_noswizzle:.3f}x, "
                f"step3_ratio(double_swizzle_manual/double_swizzle_auto)={step3_manual_vs_auto:.3f}x"
            )
        print("")

    avg_step1 = sum(ratios_step1_double_vs_single_noswizzle) / len(ratios_step1_double_vs_single_noswizzle)
    min_step1 = min(ratios_step1_double_vs_single_noswizzle)
    max_step1 = max(ratios_step1_double_vs_single_noswizzle)
    avg_step2 = sum(ratios_step2_swizzle_vs_noswizzle) / len(ratios_step2_swizzle_vs_noswizzle)
    min_step2 = min(ratios_step2_swizzle_vs_noswizzle)
    max_step2 = max(ratios_step2_swizzle_vs_noswizzle)
    avg_step3 = sum(ratios_step3_manual_vs_auto_swizzle) / len(ratios_step3_manual_vs_auto_swizzle)
    min_step3 = min(ratios_step3_manual_vs_auto_swizzle)
    max_step3 = max(ratios_step3_manual_vs_auto_swizzle)

    print("=== Summary ===")
    print("Step1 (double-buffer speedup, both non-swizzle auto-sync):")
    print(f"avg FLOP ratio(double_noswizzle_auto/single_noswizzle): {avg_step1:.3f}x")
    print(f"min FLOP ratio(double_noswizzle_auto/single_noswizzle): {min_step1:.3f}x")
    print(f"max FLOP ratio(double_noswizzle_auto/single_noswizzle): {max_step1:.3f}x")
    print("Step2 (swizzle speedup, both double-buffer auto-sync):")
    print(f"avg FLOP ratio(double_swizzle_auto/double_noswizzle_auto): {avg_step2:.3f}x")
    print(f"min FLOP ratio(double_swizzle_auto/double_noswizzle_auto): {min_step2:.3f}x")
    print(f"max FLOP ratio(double_swizzle_auto/double_noswizzle_auto): {max_step2:.3f}x")
    print("Step3 (manual-sync speedup, both double-buffer swizzle):")
    print(f"avg FLOP ratio(double_swizzle_manual/double_swizzle_auto): {avg_step3:.3f}x")
    print(f"min FLOP ratio(double_swizzle_manual/double_swizzle_auto): {min_step3:.3f}x")
    print(f"max FLOP ratio(double_swizzle_manual/double_swizzle_auto): {max_step3:.3f}x")

    _maybe_plot(plot_rows, plot_dir)


if __name__ == "__main__":
    main()
