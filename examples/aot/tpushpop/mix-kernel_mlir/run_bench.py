import argparse
import ctypes
import csv
import os
import shutil
import subprocess
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from ptodsl.npu_info import get_test_device

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(THIS_DIR, "build_artifacts")
DEFAULT_LIB_PATH = os.path.join(ARTIFACT_DIR, "tpushpop_mlir_lib.so")
DEFAULT_COMPILE_SCRIPT = os.path.join(THIS_DIR, "compile.sh")

M = 16
N = 16
TILE_BYTES = M * N * 4
FIFO_SLOTS = 8
FIFO_BYTES = TILE_BYTES * FIFO_SLOTS
ATOL = 1e-4
RTOL = 1e-4
SPLIT_VARIANTS = {
    "up-down": {
        "mode": "c2v_add_bench",
        "label": "up/down",
        "tile_fragment": "8x16 row half",
    },
    "left-right": {
        "mode": "c2v_add_bench_lr",
        "label": "left/right",
        "tile_fragment": "16x8 column half",
    },
}


@dataclass
class BenchRecord:
    tiles: int
    avg_us: float
    tile_us: float
    logical_pipe_gbps: float
    estimated_total_gbps: float
    max_abs: float


def torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def compile_for_tiles(tile_count: int, compile_script: str, mode: str) -> str:
    env = dict(
        os.environ,
        TPUSHPOP_MODE=mode,
        TPUSHPOP_BENCH_TILES=str(tile_count),
    )
    subprocess.run(["bash", compile_script], check=True, cwd=THIS_DIR, env=env)

    unique_lib = os.path.join(
        ARTIFACT_DIR,
        f"tpushpop_mlir_{mode}_tiles{tile_count}.so",
    )
    if os.path.exists(unique_lib):
        os.remove(unique_lib)
    shutil.copy2(DEFAULT_LIB_PATH, unique_lib)
    return unique_lib


def load_lib(lib_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.call_kernel.restype = None
    return lib


def make_gm_slot_buffer(*, device: str) -> torch.Tensor:
    fifo_elems = FIFO_BYTES // torch.empty((), dtype=torch.float32).element_size()
    return torch.zeros((fifo_elems,), dtype=torch.float32, device=device)


def make_io_tensors(
    tile_count: int, *, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand((tile_count, M, N), dtype=torch.float32, device=device) - 0.5
    y = torch.zeros((tile_count, M, N), dtype=torch.float32, device=device)
    return x.contiguous(), y.contiguous()


def reference(x: torch.Tensor) -> torch.Tensor:
    return 2 * torch.matmul(x.cpu(), x.cpu())


def launch_kernel(
    lib: ctypes.CDLL,
    *,
    gm_slot_buffer: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    stream_ptr,
) -> None:
    lib.call_kernel(
        1,
        stream_ptr,
        torch_to_ctypes(gm_slot_buffer),
        torch_to_ctypes(x),
        torch_to_ctypes(y),
    )


def validate(
    lib: ctypes.CDLL,
    *,
    gm_slot_buffer: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    stream_ptr,
) -> float:
    y.zero_()
    launch_kernel(
        lib,
        gm_slot_buffer=gm_slot_buffer,
        x=x,
        y=y,
        stream_ptr=stream_ptr,
    )
    torch.npu.synchronize()

    y_ref = reference(x)
    y_cpu = y.cpu()
    max_abs = float(torch.max(torch.abs(y_cpu - y_ref)).item())
    if not torch.allclose(y_cpu, y_ref, atol=ATOL, rtol=RTOL):
        raise RuntimeError(
            f"validation failed for tiles={x.shape[0]} "
            f"with max_abs={max_abs:.6f}, atol={ATOL}, rtol={RTOL}"
        )
    return max_abs


def time_kernel_seconds(
    launch_once,
    *,
    warmup: int,
    iters: int,
    inner_repeats: int,
) -> float:
    for _ in range(warmup):
        for _ in range(inner_repeats):
            launch_once()
    torch.npu.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(inner_repeats):
            launch_once()
        end.record()
        torch.npu.synchronize()
        timings.append((start.elapsed_time(end) * 1e-3) / inner_repeats)
    return sum(timings) / len(timings)


def benchmark_one(
    *,
    tile_count: int,
    lib_path: str,
    device: str,
    warmup: int,
    iters: int,
    inner_repeats: int,
) -> BenchRecord:
    lib = load_lib(lib_path)
    gm_slot_buffer = make_gm_slot_buffer(device=device)
    x, y = make_io_tensors(tile_count, device=device)
    stream_ptr = torch.npu.current_stream()._as_parameter_

    max_abs = validate(
        lib,
        gm_slot_buffer=gm_slot_buffer,
        x=x,
        y=y,
        stream_ptr=stream_ptr,
    )

    def launch_once():
        launch_kernel(
            lib,
            gm_slot_buffer=gm_slot_buffer,
            x=x,
            y=y,
            stream_ptr=stream_ptr,
        )

    avg_s = time_kernel_seconds(
        launch_once,
        warmup=warmup,
        iters=iters,
        inner_repeats=inner_repeats,
    )

    logical_pipe_bytes = tile_count * TILE_BYTES
    # Estimate visible traffic: GM read X + FIFO write + FIFO read + GM write Y.
    estimated_total_bytes = tile_count * TILE_BYTES * 4
    return BenchRecord(
        tiles=tile_count,
        avg_us=avg_s * 1e6,
        tile_us=(avg_s / tile_count) * 1e6,
        logical_pipe_gbps=(logical_pipe_bytes / avg_s) / 1e9,
        estimated_total_gbps=(estimated_total_bytes / avg_s) / 1e9,
        max_abs=max_abs,
    )


def write_csv(records: list[BenchRecord], csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tiles",
                "avg_us",
                "tile_us",
                "logical_pipe_gbps",
                "estimated_total_gbps",
                "max_abs",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.tiles,
                    f"{rec.avg_us:.6f}",
                    f"{rec.tile_us:.6f}",
                    f"{rec.logical_pipe_gbps:.6f}",
                    f"{rec.estimated_total_gbps:.6f}",
                    f"{rec.max_abs:.6f}",
                ]
            )


def plot_records(records: list[BenchRecord], plot_path: str, split_info: dict) -> None:
    if plt is None:
        print("matplotlib is not installed; skipping plot generation.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    tiles = [rec.tiles for rec in records]
    pipe_gbps = [rec.logical_pipe_gbps for rec in records]
    total_gbps = [rec.estimated_total_gbps for rec in records]
    tile_us = [rec.tile_us for rec in records]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        f"C2V Matmul + Vector Add FIFO Benchmark ({split_info['label']} split)",
        fontsize=14,
    )

    ax0.plot(tiles, pipe_gbps, "o-", label="Logical C2V pipe GB/s")
    ax0.plot(tiles, total_gbps, "s--", label="Estimated total GM GB/s")
    ax0.axvline(FIFO_SLOTS, color="0.35", linestyle=":", linewidth=1.5)
    ax0.annotate(
        "8 FIFO slots",
        xy=(FIFO_SLOTS, max(pipe_gbps) if pipe_gbps else 0),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color="0.25",
    )
    ax0.set_ylabel("GB/s")
    ax0.grid(True, linestyle="--", alpha=0.35)
    ax0.legend(loc="best")

    ax1.plot(tiles, tile_us, "o-", color="#1f7a5c", label="Time per tile")
    ax1.axvline(FIFO_SLOTS, color="0.35", linestyle=":", linewidth=1.5)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(tiles)
    ax1.set_xticklabels([str(v) for v in tiles])
    ax1.set_xlabel("16x16 tiles per kernel launch")
    ax1.set_ylabel("us / tile")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best")

    note = (
        "Each tile transfers one 1024-byte accumulator through C2V. "
        f"Each vector subcore receives one {split_info['tile_fragment']}. "
        "The FIFO backing store is fixed at 8 slots, so larger launches stream "
        "through the same queue instead of allocating more FIFO memory."
    )
    fig.text(0.02, 0.015, note, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(plot_path, dpi=160)
    print(f"Saved plot to {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and benchmark the tiled c2v_add_bench TPUSH/TPOP example."
        )
    )
    parser.add_argument(
        "--tile-counts",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="Compile-time tile counts to sweep.",
    )
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_VARIANTS),
        default="up-down",
        help="Tile split variant to benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=10,
        help="Kernel launches per timing event. Increase for more stable tiny-kernel timings.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Reuse previously compiled per-tile-count shared libraries.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Plot output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(tile_count <= 0 for tile_count in args.tile_counts):
        raise SystemExit("--tile-counts must all be positive.")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    split_info = SPLIT_VARIANTS[args.split]
    mode = split_info["mode"]
    csv_path = args.csv or os.path.join(ARTIFACT_DIR, f"{mode}.csv")
    plot_path = args.plot or os.path.join(ARTIFACT_DIR, f"{mode}.png")

    device = get_test_device()
    torch.npu.set_device(device)

    print("C2V add FIFO benchmark")
    print(
        f"split={args.split} ({split_info['label']}); "
        f"vector fragment={split_info['tile_fragment']}"
    )
    print(f"tile_shape={M}x{N} dtype=float32 tile_bytes={TILE_BYTES}")
    print(
        f"fifo_slots={FIFO_SLOTS} fifo_bytes={FIFO_BYTES} "
        f"(enough for {FIFO_SLOTS} in-flight full tiles)"
    )
    print(
        "logical_pipe_gbps counts only C2V payload bytes; "
        "estimated_total_gbps counts X read + FIFO write/read + Y write."
    )

    records = []
    for tile_count in args.tile_counts:
        if args.skip_compile:
            lib_path = os.path.join(
                ARTIFACT_DIR,
                f"tpushpop_mlir_{mode}_tiles{tile_count}.so",
            )
            if not os.path.exists(lib_path):
                raise FileNotFoundError(
                    f"{lib_path} does not exist; rerun without --skip-compile."
                )
        else:
            lib_path = compile_for_tiles(tile_count, DEFAULT_COMPILE_SCRIPT, mode)

        rec = benchmark_one(
            tile_count=tile_count,
            lib_path=lib_path,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            inner_repeats=args.inner_repeats,
        )
        records.append(rec)
        print(
            f"tiles={rec.tiles:4d} | {rec.avg_us:9.2f} us/launch | "
            f"{rec.tile_us:8.3f} us/tile | "
            f"pipe={rec.logical_pipe_gbps:8.3f} GB/s | "
            f"estimated_total={rec.estimated_total_gbps:8.3f} GB/s | "
            f"max_abs={rec.max_abs:.6f}"
        )

    write_csv(records, csv_path)
    print(f"Saved CSV to {csv_path}")
    plot_records(records, plot_path, split_info)


if __name__ == "__main__":
    main()
