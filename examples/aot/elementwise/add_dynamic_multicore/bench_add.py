import ctypes
import torch
import torch_npu
from ptodsl.test_util import get_test_device


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def lib_to_func(lib):
    def add_func(x, y, z, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        lib.call_kernel(
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(z),
            x.numel(),
        )

    return add_func


def bench_add(add_func, x, y, z, kernel_name="add_func", warmup_iters=5, benchmark_iters=50):
    io_bytes = x.numel() * x.element_size() * 3
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

    custom_ms = time_op(lambda: add_func(x, y, z))
    torch_add_ms = time_op(lambda: torch.add(x, y, out=z))

    custom_bw_gbs = (io_bytes / (custom_ms / 1e3)) / 1e9
    torch_add_bw_gbs = (io_bytes / (torch_add_ms / 1e3)) / 1e9

    print(
        f"{kernel_name}: {custom_ms:.3f} ms, "
        f"effective bandwidth: {custom_bw_gbs:.3f} GB/s "
        f"(IO={io_bytes / 1e6:.2f} MB)"
    )
    print(
        f"torch.add: {torch_add_ms:.3f} ms, "
        f"effective bandwidth: {torch_add_bw_gbs:.3f} GB/s "
        f"(IO={io_bytes / 1e6:.2f} MB)"
    )


def run_bench(lib_path, kernel_name):
    device = get_test_device()
    torch.npu.set_device(device)

    lib = ctypes.CDLL(lib_path)
    add_func = lib_to_func(lib)

    num_cores = 24 * 2  # match kernel num cores * 2
    tile_size = 8192  # match kernel tile size
    num_rounds = 100  # each core iterate this many times
    tile_count = num_rounds * num_cores
    shape = tile_size * tile_count

    torch.manual_seed(0)
    dtype = torch.float32
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    add_func(x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(z, x + y)

    bench_add(add_func, x, y, z, kernel_name=kernel_name)


if __name__ == "__main__":
    run_bench("./add_lib.so", "add_func")
    run_bench("./add_double_lib.so", "add_double_func")
