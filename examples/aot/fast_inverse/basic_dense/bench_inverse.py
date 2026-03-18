import argparse
import ctypes
import math
import random

import numpy as np
import torch
import torch_npu  # noqa: F401

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from ptodsl import do_bench
from ptodsl.test_util import get_test_device

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

SUPPORTED_MATRIX_SIZES = (16, 32, 64, 128)
DEFAULT_BATCH_SIZES = [2**k for k in range(4, 16)]  # 16, 32, ..., 32768
try:
    PERSISTENT_BLOCK_DIM = int(torch.npu.get_device_properties("npu").cube_core_num)
except Exception:
    PERSISTENT_BLOCK_DIM = 24


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def inverse_io_bytes(in_delta: torch.Tensor, out: torch.Tensor) -> int:
    # Requested traffic model: read in_delta + write out only.
    return in_delta.numel() * _dtype_nbytes(in_delta.dtype) + out.numel() * _dtype_nbytes(
        out.dtype
    )


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim (fixed core count)
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # in_delta (M - I)
        ctypes.c_void_p,  # identity_neg
        ctypes.c_uint32,  # runtime batch_size
        ctypes.c_uint32,  # log2(matrix_size)
    ]
    lib.call_kernel.restype = None
    return lib


def dense_stable_matrix(n, batch, scale=0.02):
    eye = np.eye(n, dtype=np.float32)
    noise = np.random.uniform(-1.0, 1.0, size=(batch, n, n)).astype(np.float32)
    out = eye[None, :, :] + scale * noise
    return torch.from_numpy(out)


def benchmark_kernel_seconds(kernel_launch_fn, warmup: int, iters: int) -> float:
    # Measure kernel launch only (preparation is done outside this function).
    return do_bench(kernel_launch_fn, warmup_iters=warmup, benchmark_iters=iters, unit="s")


def run_benchmark(
    lib,
    *,
    label: str,
    matrix_size: int,
    batch_sizes: list[int],
    warmup: int,
    iters: int,
):
    log2_blocksize = int(math.log2(matrix_size))
    stream_ptr = torch.npu.current_stream()._as_parameter_
    bandwidth_gib_s = []

    print(f"\n=== benchmark {label} ===")
    for batch in batch_sizes:
        inp = dense_stable_matrix(n=matrix_size, batch=batch).to(device)
        inp_fp16 = inp.to(torch.float16).contiguous()

        # Preparation work excluded from benchmark timing.
        identity = torch.eye(matrix_size, dtype=torch.float16, device=device)
        in_delta = (inp_fp16 - identity).contiguous()
        identity_neg = (-identity).contiguous()
        out = torch.zeros_like(inp_fp16, dtype=torch.float32, device=device)

        def launch_only():
            lib.call_kernel(
                PERSISTENT_BLOCK_DIM,
                stream_ptr,
                torch_to_ctypes(out),
                torch_to_ctypes(in_delta),
                torch_to_ctypes(identity_neg),
                batch,
                log2_blocksize,
            )

        avg_s = benchmark_kernel_seconds(launch_only, warmup=warmup, iters=iters)
        io_bytes = inverse_io_bytes(in_delta, out)
        total_traffic_gib = io_bytes / (1024**3)
        gib_s = io_bytes / avg_s / (1024**3)
        bandwidth_gib_s.append(gib_s)
        print(
            f"{label:>6s} | batch={batch:5d} | {avg_s * 1e3:.3f} ms | "
            f"{gib_s:.2f} GiB/s | traffic={total_traffic_gib:.4f} GiB"
        )

    return bandwidth_gib_s


def plot_results(
    batch_sizes: list[int],
    single_bw_gib_s: list[float],
    double_bw_gib_s: list[float],
    out_png: str,
    n: int,
) -> None:
    if plt is None:
        print("Warning: matplotlib is not installed; skipping plot generation.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, single_bw_gib_s, "o-", label="single buffer")
    plt.plot(batch_sizes, double_bw_gib_s, "s-", label="double buffer")
    plt.xlabel("Batch size")
    plt.ylabel("Bandwidth (GiB/s)")
    plt.title(f"Fast Inverse Bandwidth Comparison (n={n})")
    plt.xscale("log", base=2)
    plt.xticks(batch_sizes, [str(x) for x in batch_sizes])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-size",
        type=int,
        choices=SUPPORTED_MATRIX_SIZES,
        default=64,
        help="Dense matrix size n.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations for each batch size.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of measured iterations for each batch size.",
    )
    parser.add_argument(
        "--lib-path",
        type=str,
        default="./inverse_lib.so",
        help="Single-buffer shared library path produced by compile.sh.",
    )
    parser.add_argument(
        "--double-lib-path",
        type=str,
        default="./inverse_lib_db.so",
        help="Double-buffer shared library path produced by compile.sh.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="bench_inverse_bandwidth.png",
        help="Output image path for the benchmark figure.",
    )
    args = parser.parse_args()

    device = get_test_device()
    torch.npu.set_device(device)

    single_lib = load_lib(args.lib_path)
    double_lib = load_lib(args.double_lib_path)

    bw_single = run_benchmark(
        single_lib,
        label="single",
        matrix_size=args.matrix_size,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        iters=args.iters,
    )
    bw_double = run_benchmark(
        double_lib,
        label="double",
        matrix_size=args.matrix_size,
        batch_sizes=args.batch_sizes,
        warmup=args.warmup,
        iters=args.iters,
    )

    print("\n=== speedup (double / single) ===")
    for batch, single_bw, double_bw in zip(args.batch_sizes, bw_single, bw_double):
        speedup = double_bw / single_bw if single_bw > 0 else float("nan")
        print(
            f"batch={batch:5d} | single={single_bw:8.2f} GiB/s | "
            f"double={double_bw:8.2f} GiB/s | speedup={speedup:.3f}x"
        )

    plot_results(args.batch_sizes, bw_single, bw_double, args.out_png, args.matrix_size)
