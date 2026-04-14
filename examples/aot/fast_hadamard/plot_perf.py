import os
import csv

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _load_csv(csv_path):
    """Return {n: {batch: bw}} from a benchmark CSV, or {} if missing."""
    data = {}
    if not os.path.exists(csv_path):
        return data
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data.setdefault(int(row["N"]), {})[int(row["batch"])] = float(
                row["bandwidth_gbs"]
            )
    return data


def _plot_data(ax, data, block_dim):
    for idx, hidden_dim in enumerate(sorted(data.keys())):
        batches = sorted(data[hidden_dim].keys())
        bws = [data[hidden_dim][b] for b in batches]
        marker = "o" if idx < 10 else ["s", "^", "D"][idx - 10]
        ax.plot(
            batches, bws, marker=marker, markersize=4, label=f"hidden_dim={hidden_dim}"
        )

    BENCH_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]
    ax.set_xscale("log", base=2)
    ax.set_xticks(BENCH_BATCHES)
    ax.set_xticklabels([str(b) for b in BENCH_BATCHES], rotation=45, fontsize=7)
    ax.set_xlabel("batch")
    ax.set_title(
        f"BLOCK_DIM={block_dim}" if data else f"BLOCK_DIM={block_dim} (no data)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)


BENCH_BLOCK_DIMS = [20, 24]


def _make_figure(title, input_dir, csv_prefix, output_path):
    if plt is None:
        print("Warning: matplotlib is not installed; skipping plot generation.")
        return

    datasets = {
        bd: _load_csv(os.path.join(input_dir, f"{csv_prefix}{bd}.csv"))
        for bd in BENCH_BLOCK_DIMS
    }
    if not any(datasets.values()):
        return

    fig, axes = plt.subplots(1, len(BENCH_BLOCK_DIMS), figsize=(14, 6), sharey=True)
    if len(BENCH_BLOCK_DIMS) == 1:
        axes = [axes]

    for ax, block_dim in zip(axes, BENCH_BLOCK_DIMS):
        _plot_data(ax, datasets[block_dim], block_dim)

    axes[0].set_ylabel("Bandwidth (GB/s)")
    fig.suptitle(title)
    fig.tight_layout()
    out = os.path.join(input_dir, output_path)
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


def plot_bandwidth(input_dir="./perf_data/", output_path="bw_vs_shape.png"):
    """Bandwidth plot for the plain fast-Hadamard kernel (auto-sync)."""
    _make_figure(
        "Fast Hadamard PTO-DSL: Bandwidth vs Shape",
        input_dir,
        "fht_pto_bd",
        output_path,
    )


def plot_bandwidth_manual(
    input_dir="./perf_data/", output_path="bw_vs_shape_manual.png"
):
    """Bandwidth plot for the plain fast-Hadamard kernel (manual-sync)."""
    _make_figure(
        "Fast Hadamard PTO-DSL (manual-sync): Bandwidth vs Shape",
        input_dir,
        "fht_pto_manual_bd",
        output_path,
    )


def plot_bandwidth_quant(input_dir="./perf_data/", output_path="bw_vs_shape_quant.png"):
    """Bandwidth plot for the fused Hadamard+quantize kernel (auto-sync, fp16 → int8)."""
    _make_figure(
        "Fast Hadamard+Quant PTO-DSL: Bandwidth vs Shape",
        input_dir,
        "fht_quant_pto_bd",
        output_path,
    )


def plot_bandwidth_quant_manual(
    input_dir="./perf_data/", output_path="bw_vs_shape_quant_manual.png"
):
    """Bandwidth plot for the fused Hadamard+quantize kernel (manual-sync, fp16 → int8)."""
    _make_figure(
        "Fast Hadamard+Quant PTO-DSL (manual-sync): Bandwidth vs Shape",
        input_dir,
        "fht_quant_pto_manual_bd",
        output_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="./perf_data/")
    args = parser.parse_args()

    plot_bandwidth(input_dir=args.input_dir)
    plot_bandwidth_manual(input_dir=args.input_dir)
    plot_bandwidth_quant(input_dir=args.input_dir)
    plot_bandwidth_quant_manual(input_dir=args.input_dir)
