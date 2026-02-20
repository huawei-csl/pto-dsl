import ctypes
import time
import torch
import torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _dtype_nbytes(dtype: torch.dtype) -> int:
    # torch.dtype doesn't expose itemsize directly.
    return torch.empty((), dtype=dtype).element_size()


def matmul_flops(batch_size: int, m: int, k: int, n: int) -> int:
    # GEMM: C[m,n] = A[m,k] @ B[k,n] => 2*m*k*n FLOPs per batch.
    return 2 * batch_size * m * k * n


def matmul_io_bytes(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> int:
    # Simple traffic model: read A + read B + write C.
    # Does not include cache effects or intermediate buffers.
    elt = _dtype_nbytes(a.dtype)
    #return (a.numel() + b.numel() + c.numel()) * elt
    return (a.numel() + b.numel())  * elt


def _sync_if_needed(device: str) -> None:
    if device.startswith("npu"):
        torch.npu.synchronize()


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
    for _ in range(warmup):
        fn()
    _sync_if_needed(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync_if_needed(device)
    t1 = time.perf_counter()

    total_s = t1 - t0
    avg_s = total_s / iters
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


def test_matmul(verbose=False):
    device = "npu:6"
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    for blk in range(1,25):
        bs = 24*20
        m, k, n = 128, 128, 128
        torch.manual_seed(0)
        a = torch.rand((bs, m, k), device=device, dtype=dtype)
        b = torch.rand((k, n), device=device, dtype=dtype)
        c = torch.zeros((bs, m, n), device=device, dtype=dtype)

        matmul_func = load_lib("./matmul_kernel.so")
        torch.npu.synchronize()

        c_ref = torch.matmul(a, b)
        diff = (c - c_ref).abs().max()
        print('max diff: ', diff)

        flops = matmul_flops(bs, m, k, n)
        io_bytes = matmul_io_bytes(a, b, c)

        print_benchmark(
            benchmark(
                "custom_kernel",
                lambda: matmul_func(c, a, b, batch_size=a.shape[0], block_dim=blk),
                device=device,
                warmup=0,
                iters=20,
                flops=flops,
                io_bytes=io_bytes,
            )
        )
        print_benchmark(
            benchmark(
                "torch.matmul",
                lambda: torch.matmul(a, b, out=c),
                device=device,
                warmup=20,
                iters=100,
                flops=flops,
                io_bytes=io_bytes,
            )
        )


    if verbose:
        print('ref')
        print(c_ref)
        print('our')
        print(c)
        tol = 1e-3
        correct = (c - c_ref).abs() <= tol

        batch_size, m_dim, n_dim = c.shape
        step_m = 4
        step_n = 4

        for bi in range(batch_size):
            print(f"\nBatch {bi}:")
            for i in range(0, m_dim, step_m):
                for j in range(0, n_dim, step_n):
                    if correct[bi, i : i + step_m, j : j + step_n].all():
                        print("X", end="")
                    else:
                        print(".", end="")
                print("|")


import matplotlib.pyplot as plt

def plot_benchmark():
    device = "npu:6"
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    blk_values = list(range(1, 41))
    pto_results, torch_results = [], []

    matmul_func = load_lib("./matmul_kernel.so")  # assume defined
    torch.manual_seed(0)

    bs, m, k, n = 24*200, 128, 128, 128
    for blk in blk_values:

        b_percore = (bs+blk-1)//blk
        cores_needed = (bs+b_percore-1)//b_percore

        print(f'cores {blk} and cores used {cores_needed}')




        a = torch.rand((bs, m, k), device=device, dtype=dtype)
        b = torch.rand((k, n), device=device, dtype=dtype)
        c = torch.zeros((bs, m, n), device=device, dtype=dtype)

        # correctness check
        matmul_func(c, a, b, batch_size=bs, block_dim=blk)
        torch.npu.synchronize()
        assert (c - torch.matmul(a, b)).abs().max() <= 1e-5

        flops = matmul_flops(bs, m, k, n)
        io_bytes = matmul_io_bytes(a, b, c)

        # benchmarks
        torch_b = benchmark("torch.matmul",
                            lambda: torch.matmul(a, b, out=c),
                            device=device, warmup=20, iters=100,
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
    plt.plot(blk_values, torch_results, 's-', label='torch.matmul')
    plt.xlabel('Number of cores')
    plt.ylabel('Bandwidth (GB/s)')
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
#    test_matmul()
    plot_benchmark()
