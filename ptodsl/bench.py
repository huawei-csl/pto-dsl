from typing import Callable, List, Literal, Union


def do_bench(
    fn: Callable,
    warmup_iters: int = 5,
    benchmark_iters: int = 15,
    aggregation: Literal["mean", "none"] = "mean",
    unit: Literal["s", "ms", "us", "ns"] = "us",
    flush_cache: bool = True,
) -> Union[float, List[float]]:
    """
    Benchmark a given function with warmup.

    Args:
        fn: Function to benchmark.
        warmup_iters: Number of warmup runs.
        benchmark_iters: Number of benchmark runs.
        aggregation: Aggregation mode for benchmark times.
        unit: Time unit of the benchmarks.
        flush_cache: if we should overwrite l2 cache between every iteration
    Returns:
        Runtime, or list of runtimes, in specified units.
    """
    import torch
    import torch_npu
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(benchmark_iters)]

    # Allocate a 256 MB tensor which we write to every iteration to flush L2 cache
    # https://github.com/tile-ai/tilelang/blob/main/tilelang/profiler/bench.py#L103
    cache_size = 256 * 1024 * 1024
    cache = torch.empty((cache_size), dtype=torch.int8).npu()

    for _ in range(warmup_iters):
        fn()
    torch_npu.npu.synchronize()

    for i in range(benchmark_iters):
        if flush_cache:
            cache.zero_()
        torch_npu.npu.synchronize()
        start_events[i].record()
        fn()
        end_events[i].record()
        torch_npu.npu.synchronize()

    f = {"s": 1e-3, "ms": 1e0, "us": 1e3, "ns": 1e6}[unit]
    times = [f * s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    if aggregation == "mean":
        return sum(times) / len(times)
    return times
