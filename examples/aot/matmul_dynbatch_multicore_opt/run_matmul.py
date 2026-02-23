import ctypes
import time
import torch
import torch_npu
import matplotlib.pyplot as plt

from ptodsl import do_bench


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def matmul_flops(batch_size: int, m: int, k: int, n: int) -> int:
    # GEMM: C[m,n] = A[m,k] @ B[k,n] => 2*m*k*n FLOPs per batch.
    return 2 * batch_size * m * k * n


def matmul_io_bytes(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> int:
    # Simple traffic model: read A + read B + write C.
    # Does not include cache effects or intermediate buffers.
    elt = _dtype_nbytes(a.dtype)
    return (a.numel() + b.numel() + c.numel()) * elt
    #return (a.numel() + b.numel())  * elt




def benchmark(
    name: str,
    fn,
    *,
    device: str,
    warmup: int = 10,
    iters: int = 100,
    flops: int | None = None,
    io_bytes: int | None = None,
) -> dict:
    avg_s = do_bench(fn, warmup_iters=warmup, benchmark_iters=iters, unit='s')
    stats = {"name": name, "iters": iters, "avg_ms": avg_s * 1e3}
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

    def matmul_func(
        c, a, b, batch_size,
        block_dim,
        stream_ptr=None
    ):
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
    device = "npu:7"
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    blk_values = list(range(1, 25))
    pto_results, torch_results = [], []

    matmul_func = load_lib("./matmul_kernel.so")  # assume defined
    torch.manual_seed(0)

    bs, m, k, n = 24*200, 128, 128, 128
    for blk in blk_values:
        a = torch.rand((bs, m, k), device=device, dtype=dtype)
        b = torch.rand((k, n), device=device, dtype=dtype)
        c = torch.zeros((bs, m, n), device=device, dtype=dtype)

        # correctness check
        matmul_func(c, a, b, batch_size=bs, block_dim=blk)
        torch.npu.synchronize()
        c_ref = torch.matmul(a, b)
        diff = (c - c_ref).abs().max()
        assert  diff <= 1e-5, diff

        flops = matmul_flops(bs, m, k, n)
        io_bytes = matmul_io_bytes(a, b, c)

        torch_b = benchmark("torch.matmul",
                            lambda: torch.matmul(a, b, out=c),
                            device=device, warmup=20, iters=20,
                            flops=flops, io_bytes=io_bytes)['gbps']
        pto = benchmark("custom_kernel",
                        lambda: matmul_func(c, a, b, batch_size=bs, block_dim=blk),
                        device=device, warmup=20, iters=20,
                        flops=flops, io_bytes=io_bytes)['gbps']

        pto_results.append(pto)
        torch_results.append(torch_b)

    total_mb = 4 * (a.numel() + b.numel() + c.numel()) / 10**6
    # plot results
    plt.figure(figsize=(8,5))
    plt.plot(blk_values, pto_results, 'o-', label='mlir')
    plt.plot(blk_values, torch_results, 's-', label='torch.matmul (all cores)')
    plt.xlabel('Number of cores')
    plt.ylabel('Bandwidth (Read A+B write C) (GB/s)')
    plt.title(
        f"""Benchmark: Custom Kernel vs torch.matmul\n
         A: {tuple(a.shape)} B: {tuple(b.shape)}, C: {tuple(c.shape)} \n
         A+B+C size: {total_mb:.1f} MB"""
    )
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('our.png')

if __name__ == "__main__":
    plot_benchmark()
