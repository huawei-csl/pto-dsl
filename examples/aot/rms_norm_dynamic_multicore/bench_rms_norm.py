import argparse
import ctypes

import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, block_dim=24):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x
        ctypes.c_void_p,  # w
        ctypes.c_void_p,  # y (output)
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # n_cols
    ]
    lib.call_kernel.restype = None

    def rms_norm_func(x, w, y, batch, n_cols, block_dim=block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(w),
            torch_to_ctypes(y),
            batch,
            n_cols,
        )

    return rms_norm_func


def bench_rms_norm(
    rms_norm_func, x, w, y, kernel_name="rms_norm_func", warmup_iters=5, benchmark_iters=50
):
    batch, n_cols = x.shape
    # reads x and w, writes y  (w is small: n_cols << batch*n_cols for large batch)
    io_bytes = x.numel() * x.element_size() * 2 + w.numel() * w.element_size()
    # Overwrite a large buffer between launches to reduce L2 cache reuse.
    cache = torch.empty((256 * 1024 * 1024,), dtype=torch.int8, device=x.device)

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

    custom_ms = time_op(lambda: rms_norm_func(x, w, y, batch, n_cols))
    torch_ms = time_op(
        lambda: torch.nn.functional.rms_norm(x, (n_cols,), w)
    )

    custom_bw_gbs = (io_bytes / (custom_ms / 1e3)) / 1e9
    torch_bw_gbs = (io_bytes / (torch_ms / 1e3)) / 1e9

    print(
        f"{kernel_name}: {custom_ms:.3f} ms, "
        f"effective bandwidth: {custom_bw_gbs:.3f} GB/s "
        f"(IO={io_bytes / 1e6:.2f} MB)"
    )
    print(
        f"torch.rms_norm: {torch_ms:.3f} ms, "
        f"effective bandwidth: {torch_bw_gbs:.3f} GB/s "
        f"(IO={io_bytes / 1e6:.2f} MB)"
    )


def run_bench(lib_path, block_dim=24, batch=1024, n_cols=4096):
    device = get_test_device()
    torch.npu.set_device(device)

    rms_norm = load_lib(lib_path, block_dim=block_dim)

    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.randn(batch, n_cols, device=device, dtype=dtype)
    w = torch.randn(n_cols, device=device, dtype=dtype)
    y = torch.empty(batch, n_cols, device=device, dtype=dtype)

    rms_norm(x, w, y, batch, n_cols)
    torch.npu.synchronize()

    x_f32 = x.float()
    mean_sq = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
    ref = (x_f32 * torch.rsqrt(mean_sq) * w.float()).to(dtype)
    torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)

    bench_rms_norm(rms_norm, x, w, y, kernel_name=f"rms_norm ({lib_path})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="./rms_norm_lib.so")
    parser.add_argument("--block-dim", type=int, default=24)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--n-cols", type=int, default=4096)
    args = parser.parse_args()
    run_bench(args.lib, block_dim=args.block_dim, batch=args.batch, n_cols=args.n_cols)
