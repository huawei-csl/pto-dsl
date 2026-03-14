import os
import csv

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def plot_bandwidth(input_dir="./perf_data/", output_path="bw_vs_shape.png"):
    """Generate bandwidth plot from benchmark CSVs."""
    if plt is None:
        print("Warning: matplotlib is not installed; skipping plot generation.")
        return

    BENCH_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]
    BENCH_BLOCK_DIMS = [20, 24]

    fig, axes = plt.subplots(1, len(BENCH_BLOCK_DIMS), figsize=(14, 6), sharey=True)
    if len(BENCH_BLOCK_DIMS) == 1:
        axes = [axes]

    for ax, block_dim in zip(axes, BENCH_BLOCK_DIMS):
        csv_path = os.path.join(input_dir, f"fht_pto_bd{block_dim}.csv")
        if not os.path.exists(csv_path):
            ax.set_title(f"BLOCK_DIM={block_dim} (no data)")
            continue

        # Parse CSV: hidden_dim -> {batch: bw}
        data = {}
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                batch = int(row["batch"])
                n = int(row["N"])
                bw = float(row["bandwidth_gbs"])
                data.setdefault(n, {})[batch] = bw

        for idx, hidden_dim in enumerate(sorted(data.keys())):
            batches = sorted(data[hidden_dim].keys())
            bws = [data[hidden_dim][b] for b in batches]

            if idx < 10:
                marker = "o"
            else:
                last_markers = ["s", "^", "D"]
                marker = last_markers[idx - 10]

            ax.plot(
                batches,
                bws,
                marker=marker,
                markersize=4,
                label=f"hidden_dim={hidden_dim}",
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(BENCH_BATCHES)
        ax.set_xticklabels([str(b) for b in BENCH_BATCHES], rotation=45, fontsize=7)
        ax.set_xlabel("batch")
        ax.set_title(f"BLOCK_DIM={block_dim}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    axes[0].set_ylabel("Bandwidth (GB/s)")
    fig.suptitle("Fast Hadamard PTO-DSL: Bandwidth vs Shape")
    fig.tight_layout()
    fig.savefig(input_dir + output_path, dpi=150)
    print(f"\nPlot saved to {input_dir+output_path}")


if __name__ == "__main__":
    plot_bandwidth()
