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


def bench_add(add_func, x, y, z, warmup_iters=5, benchmark_iters=50):
    io_bytes = x.numel() * x.element_size() * 3

    def time_op(fn):
        for _ in range(warmup_iters):
            fn()
        torch.npu.synchronize()

        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)

        start_event.record()
        for _ in range(benchmark_iters):
            fn()
        end_event.record()
        torch.npu.synchronize()

        total_ms = start_event.elapsed_time(end_event)
        return total_ms / benchmark_iters

    custom_ms = time_op(lambda: add_func(x, y, z))
    torch_add_ms = time_op(lambda: torch.add(x, y, out=z))

    custom_bw_gbs = (io_bytes / (custom_ms / 1e3)) / 1e9
    torch_add_bw_gbs = (io_bytes / (torch_add_ms / 1e3)) / 1e9

    print(
        f"add_func: {custom_ms:.3f} ms, "
        f"effective bandwidth: {custom_bw_gbs:.3f} GB/s "
        f"(IO={io_bytes / 1e6:.2f} MB)"
    )
    print(
        f"torch.add: {torch_add_ms:.3f} ms, "
        f"effective bandwidth: {torch_add_bw_gbs:.3f} GB/s "
        f"(IO={io_bytes / 1e6:.2f} MB)"
    )


if __name__ == "__main__":
    device = get_test_device()
    torch.npu.set_device(device)

    lib = ctypes.CDLL("./add_lib.so")
    add_func = lib_to_func(lib)

    num_cores = 24 * 2
    tile_size = 1024
    num_rounds = 20  # each core iterate this many times
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

    bench_add(add_func, x, y, z)
