import argparse
import ctypes

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from ptodsl.npu_info import get_num_cube_cores, get_test_device

_DEFAULT_NUM_CORES = get_num_cube_cores()


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, block_dim=_DEFAULT_NUM_CORES):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # a
        ctypes.c_void_p,  # b
        ctypes.c_void_p,  # c (output)
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # n_cols
    ]
    lib.call_kernel.restype = None

    def geglu_func(a, b, c, batch, n_cols, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            torch_to_ctypes(c),
            batch,
            n_cols,
        )

    return geglu_func


def bench_geglu(
    geglu_func, a, b, c, kernel_name="geglu_func", warmup_iters=5, benchmark_iters=50
):
    batch, n_cols = a.shape
    # reads a and b, writes c
    io_bytes = a.numel() * a.element_size() * 3
    # Overwrite a large buffer between launches to reduce L2 cache reuse.
    cache = torch.empty((256 * 1024 * 1024,), dtype=torch.int8, device=a.device)

    def time_op(fn):
        for _ in range(warmup_iters):
            fn()
        torch.npu.synchronize()

        mixed_start = torch.npu.Event(enable_timing=True)
        mixed_end = torch.npu.Event(enable_timing=True)
        cache_start = torch.npu.Event(enable_timing=True)
        cache_end = torch.npu.Event(enable_timing=True)

        mixed_start.record()
        for _ in range(benchmark_iters):
            cache.zero_()
            fn()
        mixed_end.record()
        torch.npu.synchronize()

        cache_start.record()
        for _ in range(benchmark_iters):
            cache.zero_()
        cache_end.record()
        torch.npu.synchronize()

        mixed_total_ms = mixed_start.elapsed_time(mixed_end)
        cache_total_ms = cache_start.elapsed_time(cache_end)
        kernel_total_ms = max(mixed_total_ms - cache_total_ms, 0.0)
        return kernel_total_ms / benchmark_iters

    custom_ms = time_op(lambda: geglu_func(a, b, c, batch, n_cols))
    torch_ms = time_op(lambda: torch.mul(F.gelu(a, approximate="tanh"), b))

    custom_bw_gbs = (io_bytes / (custom_ms / 1e3)) / 1e9
    torch_bw_gbs = (io_bytes / (torch_ms / 1e3)) / 1e9

    print(
        f"{kernel_name}: {custom_ms:.3f} ms, "
        f"effective bandwidth: {custom_bw_gbs:.3f} GB/s "
        f"(IO={io_bytes / 1e6:.2f} MB)"
    )
    print(
        f"torch gelu*b: {torch_ms:.3f} ms, "
        f"effective bandwidth: {torch_bw_gbs:.3f} GB/s "
        f"(IO={io_bytes / 1e6:.2f} MB)"
    )


def run_bench(lib_path, block_dim=_DEFAULT_NUM_CORES, batch=1024, n_cols=8192):
    device = get_test_device()
    torch.npu.set_device(device)

    geglu_func = load_lib(lib_path, block_dim=block_dim)

    torch.manual_seed(0)
    dtype = torch.float16
    a = torch.randn(batch, n_cols, device=device, dtype=dtype).clamp(-4, 4)
    b = torch.randn(batch, n_cols, device=device, dtype=dtype)
    c = torch.empty(batch, n_cols, device=device, dtype=dtype)

    geglu_func(a, b, c, batch, n_cols)
    torch.npu.synchronize()

    a_f32 = a.float()
    ref = (0.5 * a_f32 * (1.0 + torch.tanh(a_f32))).to(dtype) * b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    bench_geglu(geglu_func, a, b, c, kernel_name=f"geglu ({lib_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="./geglu_lib.so")
    parser.add_argument("--block-dim", type=int, default=24)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--n-cols", type=int, default=8192)
    args = parser.parse_args()
    run_bench(args.lib, block_dim=args.block_dim, batch=args.batch, n_cols=args.n_cols)
