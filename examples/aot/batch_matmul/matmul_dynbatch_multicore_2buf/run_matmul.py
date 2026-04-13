from typing import Callable, List, Literal, Union
import ctypes
import time
import argparse

from ptodsl.npu_info import get_num_cube_cores, get_test_device
from ptodsl import do_bench

import torch
import torch_npu

_DEFAULT_NUM_CORES = get_num_cube_cores()


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def matmul_flops(batch_size: int, m: int, k: int, n: int) -> int:
    return 2 * batch_size * m * k * n


def matmul_io_bytes(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> int:
    # Simple traffic model: read A + read B + write C.
    elt = _dtype_nbytes(a.dtype)
    return (a.numel() + b.numel() + c.numel()) * elt


def benchmark(
    fn,
    *,
    flops: int | None = None,
    io_bytes: int | None = None,
) -> dict:
    avg_s = do_bench(fn, unit="s", flush_cache=True)
    stats = {"avg_ms": avg_s * 1e3}
    if flops is not None:
        stats["tflops"] = (flops / avg_s) / 1e12
    if io_bytes is not None:
        stats["gbps"] = (io_bytes / avg_s) / 1e9
    return stats


def print_benchmark(stats: dict) -> None:
    parts = [f"{stats['name']}: {stats['avg_ms']:.3f} ms"]
    if "tflops" in stats:
        parts.append(f"{stats['tflops']:.2f} TFLOP/s")
    if "gbps" in stats:
        parts.append(f"{stats['gbps']:.2f} GB/s (A+B+C)")
    print(" | ".join(parts))


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)

    def matmul_func(c, a, b, batch_size, block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(c),
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            ctypes.c_uint32(batch_size),
        )

    return matmul_func


def plot_benchmark():
    import matplotlib.pyplot as plt

    device = get_test_device()
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float16
    torch.manual_seed(0)

    matmul_func = load_lib("./matmul_kernel.so")

    pto_results, torch_results, pto2_results, pto3_results = [], [], [], []
    m, k, n = 128, 128, 128
    batches = list(range(24 * 2, 8000, 24 * 2))
    blk = [_DEFAULT_NUM_CORES, 1, 6]
    for i in batches:
        bs = i
        a = torch.rand((bs, m, k), device=device, dtype=dtype)
        b = torch.rand((k, n), device=device, dtype=dtype)
        c = torch.zeros((bs, m, n), device=device, dtype=dtype)

        # correctness check
        matmul_func(c, a, b, batch_size=bs, block_dim=_DEFAULT_NUM_CORES)
        torch.npu.synchronize()
        c_ref = torch.matmul(a, b)
        diff = (c - c_ref).abs().max()
        # assert  diff <= 1e-5, diff
        if diff < 1e-5:
            print(".", end="")
        else:
            print(f"failed at shape: {a.shape} with {diff}")

        flops = matmul_flops(bs, m, k, n)
        io_bytes = matmul_io_bytes(a, b, c)

        # run a benchmark for warmup (else first iterations are off)
        benchmark(lambda: torch.matmul(a, b, out=c))

        torch_b = benchmark(
            lambda: torch.matmul(a, b, out=c), flops=flops, io_bytes=io_bytes
        )["gbps"]
        pto2 = benchmark(
            lambda: matmul_func(c, a, b, batch_size=bs, block_dim=blk[1]),
            flops=flops,
            io_bytes=io_bytes,
        )["gbps"]
        pto3 = benchmark(
            lambda: matmul_func(c, a, b, batch_size=bs, block_dim=blk[2]),
            flops=flops,
            io_bytes=io_bytes,
        )["gbps"]
        pto = benchmark(
            lambda: matmul_func(c, a, b, batch_size=bs, block_dim=blk[0]),
            flops=flops,
            io_bytes=io_bytes,
        )["gbps"]
        pto_results.append(pto)
        pto2_results.append(pto2)
        pto3_results.append(pto3)
        torch_results.append(torch_b)
    print()
    rel_diff = [our / their for our, their in zip(pto_results, torch_results)]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(batches, pto_results, "-", label=f"pto-dsl ({blk[0]} cores)")
    ax1.plot(batches, pto2_results, "-", label=f"pto-dsl ({blk[1]} cores)")
    ax1.plot(batches, pto3_results, "-", label=f"pto-dsl ({blk[2]} cores)")
    ax1.plot(batches, torch_results, "-", label="torch.matmul (24 cores)")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Bandwidth (Read A+B write C) (GB/s)")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(batches, rel_diff, "-", color="purple", label="pto-dsl / torch")
    ax2.set_ylabel("Relative Performance (pto-dsl / torch)")
    ax2.set_ylim(0.95 * min(rel_diff), 1.05 * max(rel_diff))
    ax2.axhline(y=1, linestyle="--", linewidth=1.0)

    dt_str = {torch.float16: "fp16", torch.float32: "fp32"}[dtype]
    plt.title(
        f"""pto-dsl kernel vs torch.matmul\n
        <B, {a.shape[1]}, {a.shape[2]}, {dt_str}>@<{b.shape[0]}, {b.shape[1]}, {dt_str}>=<B, {c.shape[1]}, {c.shape[2]}, {dt_str}>"""
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.tight_layout()
    plt.savefig("dsl.png")


def correctness_verify():
    device = get_test_device()
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float16
    torch.manual_seed(0)

    matmul_func = load_lib("./matmul_kernel.so")

    m, k, n = 128, 128, 128
    for blk in [1, _DEFAULT_NUM_CORES]:
        for bs in range(1000, 1100):
            a = torch.rand((bs, m, k), device=device, dtype=dtype)
            b = torch.rand((k, n), device=device, dtype=dtype)
            c = torch.zeros((bs, m, n), device=device, dtype=dtype)

            matmul_func(c, a, b, batch_size=bs, block_dim=blk)
            torch.npu.synchronize()
            c_ref = torch.matmul(a, b)

            diff = (c - c_ref).abs().max()
            # assert  diff <= 1e-5, diff
            if diff < 1e-5:
                print(".", end="", flush=True)
            else:
                print(
                    f"#cores={blk} failed at shape: {list(a.shape)} with error:{diff}"
                )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark", dest="benchmark", action="store_true", help="Enable benchmarking"
    )
    args = parser.parse_args()
    correctness_verify()
    if args.benchmark:
        plot_benchmark()
