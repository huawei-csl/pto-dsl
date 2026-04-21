"""Benchmark fp16 Sinkhorn — torch vs PTODSL kernel vs hand-tuned reference.

Shapes:
  K (head_dim) : 64, 128, 256
  L (n_tokens) : 32, 64, 128, 256
  Batch        : 1 (one (K, L) matrix per call)

Plus a batched-vs-serial sweep at K=L=128.

Writes:
  outputs/csv/head_shapes_bench.csv
  outputs/csv/batched_vs_serial.csv
  outputs/plots/head_shapes_*.png
  outputs/plots/batched_vs_serial_log.png
"""

# pylint: disable=wrong-import-position
import argparse
import csv
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from jit_util_sinkhorn import (
    BLOCK_DIM,
    jit_compile_pto,
    jit_compile_reference,
)
from ptodsl.npu_info import get_test_device

THIS_DIR = Path(__file__).resolve().parent

# --- Sinkhorn hyperparameters ----------------------------------------------
SINKHORN_ORDER = 8
SINKHORN_LR = 0.9
SINKHORN_EPS = 1e-6

# --- Benchmark grids -------------------------------------------------------
HEAD_DIMS = [64, 128, 256]
N_TOKENS = [32, 64, 128, 256]

BATCH_SIZES = [1, 4, 8, 16, 32, 64, 128, 256]
BATCH_K = 128
BATCH_L = 128

KERNEL_WARMUP = 10
KERNEL_REPEATS = 50


# --- torch reference -------------------------------------------------------


def sinq_torch_fp16(matrix, sinkhorn_order=8, sinkhorn_lr=0.9, sinkhorn_eps=1e-6):
    """Vectorised torch SINQ on (N, K, L). Stays in fp16."""
    K, L = matrix.shape[-2], matrix.shape[-1]
    m = matrix
    mu1 = torch.ones(*matrix.shape[:-2], L, dtype=m.dtype, device=m.device)
    mu2 = torch.ones(*matrix.shape[:-2], K, 1, dtype=m.dtype, device=m.device)
    tgt = (
        torch.minimum(
            m.std(dim=-1).amin(dim=-1, keepdim=True),
            m.std(dim=-2).amin(dim=-1, keepdim=True),
        ).unsqueeze(-1)
        + sinkhorn_eps
    )
    for _ in range(sinkhorn_order):
        cur = m / mu1.unsqueeze(-2) / mu2
        mu1 = mu1 * (cur.std(dim=-2) / tgt.squeeze(-1)) ** sinkhorn_lr
        mu2 = mu2 * ((cur.std(dim=-1) / tgt.squeeze(-1)) ** sinkhorn_lr).unsqueeze(-1)
    return m / mu1.unsqueeze(-2) / mu2, mu1, mu2.squeeze(-1)


# --- timing / metric helpers ----------------------------------------------


def time_npu(fn, warmup=None, repeats=None):
    warmup = KERNEL_WARMUP if warmup is None else warmup
    repeats = KERNEL_REPEATS if repeats is None else repeats
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats  # us


def bytes_per_call(K, L, dtype_bytes):
    return (2 * K * L + L + K) * dtype_bytes


def flops_per_call(K, L, order):
    return K * L * (6 * (order + 1) + 2)


# --- head-shapes bench -----------------------------------------------------


def _call_kernel(fn, mat, out, mu1, mu2, stream_ptr):
    return fn(
        mat,
        out,
        mu1,
        mu2,
        order=SINKHORN_ORDER,
        lr=SINKHORN_LR,
        eps=SINKHORN_EPS,
        stream_ptr=stream_ptr,
    )


def run_head_shapes(pto_func, ref_func, stream_ptr, device):
    rows = []
    header = (
        f"{'K':>4} {'L':>4} | "
        f"{'torch_us':>9} {'pto_us':>8} {'ref_us':>8} | "
        f"{'pto/torch':>9} {'pto/ref':>8}"
    )
    print(header)
    print("-" * len(header))

    for K in HEAD_DIMS:
        for L in N_TOKENS:
            torch.random.manual_seed(42)
            mat = torch.rand(1, K, L, dtype=torch.float16, device=device) + 0.1
            out = torch.empty_like(mat)
            mu1 = torch.empty(1, L, dtype=torch.float16, device=device)
            mu2 = torch.empty(1, K, dtype=torch.float16, device=device)

            t_us = time_npu(
                lambda: sinq_torch_fp16(
                    mat,
                    sinkhorn_order=SINKHORN_ORDER,
                    sinkhorn_lr=SINKHORN_LR,
                    sinkhorn_eps=SINKHORN_EPS,
                )
            )
            p_us = time_npu(
                lambda: _call_kernel(pto_func, mat, out, mu1, mu2, stream_ptr)
            )
            r_us = time_npu(
                lambda: _call_kernel(ref_func, mat, out, mu1, mu2, stream_ptr)
            )

            B = bytes_per_call(K, L, 2)
            F = flops_per_call(K, L, SINKHORN_ORDER)
            sp_torch = t_us / p_us
            sp_ref = r_us / p_us
            print(
                f"{K:>4d} {L:>4d} | "
                f"{t_us:>9.2f} {p_us:>8.2f} {r_us:>8.2f} | "
                f"{sp_torch:>9.2f}x {sp_ref:>7.2f}x"
            )
            rows.append(
                {
                    "K": K,
                    "L": L,
                    "torch_us": t_us,
                    "pto_us": p_us,
                    "ref_us": r_us,
                    "torch_GB_s": B / (t_us * 1e3),
                    "pto_GB_s": B / (p_us * 1e3),
                    "ref_GB_s": B / (r_us * 1e3),
                    "torch_GFLOPS": F / (t_us * 1e3),
                    "pto_GFLOPS": F / (p_us * 1e3),
                    "ref_GFLOPS": F / (r_us * 1e3),
                    "speedup_pto_vs_torch": sp_torch,
                    "speedup_pto_vs_ref": sp_ref,
                }
            )
    return rows


# --- batched-vs-serial bench -----------------------------------------------


def run_batched_vs_serial(pto_func, ref_func, stream_ptr, device):
    print(f"\nK={BATCH_K}, L={BATCH_L}, order={SINKHORN_ORDER}")
    print(
        f"{'N':>5}  {'pto bat us':>11}  {'pto ser us':>11}  "
        f"{'ref bat us':>11}  {'pto/ref bat':>12}"
    )
    rows = []
    for N in BATCH_SIZES:
        mat = torch.rand(N, BATCH_K, BATCH_L, dtype=torch.float16, device=device) + 0.1
        out = torch.empty_like(mat)
        mu1 = torch.empty(N, BATCH_L, dtype=torch.float16, device=device)
        mu2 = torch.empty(N, BATCH_K, dtype=torch.float16, device=device)

        p_bat = time_npu(lambda: _call_kernel(pto_func, mat, out, mu1, mu2, stream_ptr))
        r_bat = time_npu(lambda: _call_kernel(ref_func, mat, out, mu1, mu2, stream_ptr))

        mats_1 = [
            (
                torch.rand(1, BATCH_K, BATCH_L, dtype=torch.float16, device=device)
                + 0.1,
                torch.empty(1, BATCH_K, BATCH_L, dtype=torch.float16, device=device),
                torch.empty(1, BATCH_L, dtype=torch.float16, device=device),
                torch.empty(1, BATCH_K, dtype=torch.float16, device=device),
            )
            for _ in range(N)
        ]

        def _serial(fn):
            for m, o, m1, m2 in mats_1:
                _call_kernel(fn, m, o, m1, m2, stream_ptr)

        p_ser = time_npu(lambda: _serial(pto_func))
        sp_pto_ref = r_bat / p_bat if p_bat > 0 else float("nan")

        print(
            f"{N:>5d}  {p_bat:>11.2f}  {p_ser:>11.2f}  "
            f"{r_bat:>11.2f}  {sp_pto_ref:>11.2f}x"
        )
        rows.append(
            {
                "N": N,
                "pto_batched_us": p_bat,
                "pto_serial_us": p_ser,
                "pto_batched_per_mat_us": p_bat / N,
                "pto_serial_per_mat_us": p_ser / N,
                "ref_batched_us": r_bat,
                "ref_batched_per_mat_us": r_bat / N,
                "speedup_pto_vs_ref": sp_pto_ref,
                "speedup_batched_vs_serial": (
                    p_ser / p_bat if p_bat > 0 else float("nan")
                ),
            }
        )
    return rows


# --- plots ----------------------------------------------------------------


def _shape_labels(rows):
    return [f"{r['K']}x{r['L']}" for r in rows]


def plot_speedup_grid(rows, key, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.full((len(HEAD_DIMS), len(N_TOKENS)), np.nan)
    for r in rows:
        i = HEAD_DIMS.index(r["K"])
        j = N_TOKENS.index(r["L"])
        grid[i, j] = r[key]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    vmax = max(np.nanmax(grid), 1.0)
    im = ax.imshow(
        grid, aspect="auto", cmap="viridis", vmin=min(np.nanmin(grid), 1.0), vmax=vmax
    )
    ax.set_xticks(range(len(N_TOKENS)), [str(l) for l in N_TOKENS])
    ax.set_yticks(range(len(HEAD_DIMS)), [str(k) for k in HEAD_DIMS])
    ax.set_xlabel("n_tokens (L)")
    ax.set_ylabel("head_dim (K)")
    ax.set_title(title)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j,
                i,
                f"{grid[i, j]:.2f}x",
                ha="center",
                va="center",
                color="white" if grid[i, j] < vmax * 0.6 else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, label="speedup (x)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved -> {path}")


def _grouped_bar(rows, keys, colors, labels, ylabel, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xlabels = _shape_labels(rows)
    x = np.arange(len(xlabels))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (key, color, label) in enumerate(zip(keys, colors, labels)):
        vals = [r[key] for r in rows]
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, vals, w, label=label, color=color)
    ax.set_xticks(x, xlabels, rotation=45, ha="right")
    ax.set_xlabel("shape (head_dim x n_tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved -> {path}")


def plot_batched(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = [r["N"] for r in rows]
    p_bat = [r["pto_batched_per_mat_us"] for r in rows]
    p_ser = [r["pto_serial_per_mat_us"] for r in rows]
    r_bat = [r["ref_batched_per_mat_us"] for r in rows]
    sp = [r["speedup_pto_vs_ref"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(Ns, p_bat, "o-", color="#dc2626", label="PTODSL batched")
    ax1.plot(Ns, p_ser, "s--", color="#94a3b8", label="PTODSL serial")
    ax1.plot(Ns, r_bat, "^-", color="#0369a1", label="reference batched")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(Ns, [str(n) for n in Ns])
    ax1.set_xlabel("batch size N")
    ax1.set_ylabel("per-matrix latency (us, log)")
    ax1.set_title(f"Per-matrix cost @ K=L={BATCH_K}")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.plot(
        Ns, [r["pto_batched_us"] for r in rows], "o-", color="#dc2626", label="PTODSL"
    )
    ax2.plot(
        Ns,
        [r["ref_batched_us"] for r in rows],
        "^-",
        color="#0369a1",
        label="reference",
    )
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(Ns, [str(n) for n in Ns])
    ax2.set_xlabel("batch size N")
    ax2.set_ylabel("total batched latency (us, log)")
    ax2.set_title("Total wall time + ref/PTO speedup")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left")
    ax2_r = ax2.twinx()
    ax2_r.plot(Ns, sp, "v:", color="#059669", label="ref/PTO speedup")
    ax2_r.set_ylabel("speedup (ref / PTO, x)", color="#059669")
    ax2_r.tick_params(axis="y", labelcolor="#059669")

    fig.suptitle("Sinkhorn — PTODSL vs reference (batched & serial)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved -> {path}")


# --- main -----------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--warmup", type=int, default=KERNEL_WARMUP)
    p.add_argument("--repeats", type=int, default=KERNEL_REPEATS)
    p.add_argument("--skip-batched", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {path}")


def main():
    global KERNEL_WARMUP, KERNEL_REPEATS
    args = _parse_args()
    KERNEL_WARMUP = args.warmup
    KERNEL_REPEATS = args.repeats

    device = get_test_device()
    torch.npu.set_device(device)
    print(f"Using device: {device}, block_dim={BLOCK_DIM}")

    print("Compiling PTODSL kernel ...")
    pto_func = jit_compile_pto(verbose=True, force=args.force_rebuild)
    print("Compiling reference.cpp ...")
    ref_func = jit_compile_reference(verbose=True, force=args.force_rebuild)

    stream_ptr = torch.npu.current_stream()._as_parameter_

    csv_dir = THIS_DIR / "outputs" / "csv"
    plot_dir = THIS_DIR / "outputs" / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- head shapes ---
    hs_rows = run_head_shapes(pto_func, ref_func, stream_ptr, device)
    _write_csv(csv_dir / "head_shapes_bench.csv", hs_rows)

    plot_speedup_grid(
        hs_rows,
        "speedup_pto_vs_torch",
        f"PTODSL vs torch fp16 — speedup (x), order={SINKHORN_ORDER}",
        plot_dir / "head_shapes_speedup_pto_vs_torch.png",
    )
    plot_speedup_grid(
        hs_rows,
        "speedup_pto_vs_ref",
        f"PTODSL vs reference C++ — speedup (x), order={SINKHORN_ORDER}",
        plot_dir / "head_shapes_speedup_pto_vs_ref.png",
    )
    _grouped_bar(
        hs_rows,
        keys=["torch_GB_s", "pto_GB_s", "ref_GB_s"],
        colors=["#94a3b8", "#dc2626", "#0369a1"],
        labels=["torch fp16", "PTODSL", "reference C++"],
        ylabel="effective bandwidth (GB/s)",
        title=f"Sinkhorn fp16 bandwidth — order={SINKHORN_ORDER}, batch=1",
        path=plot_dir / "head_shapes_bandwidth.png",
    )
    _grouped_bar(
        hs_rows,
        keys=["torch_GFLOPS", "pto_GFLOPS", "ref_GFLOPS"],
        colors=["#94a3b8", "#dc2626", "#0369a1"],
        labels=["torch fp16", "PTODSL", "reference C++"],
        ylabel="effective GFLOPS",
        title=f"Sinkhorn fp16 compute — order={SINKHORN_ORDER}, batch=1",
        path=plot_dir / "head_shapes_flops.png",
    )

    # --- batched vs serial ---
    if not args.skip_batched:
        bs_rows = run_batched_vs_serial(pto_func, ref_func, stream_ptr, device)
        _write_csv(csv_dir / "batched_vs_serial.csv", bs_rows)
        plot_batched(bs_rows, plot_dir / "batched_vs_serial_log.png")


if __name__ == "__main__":
    main()
