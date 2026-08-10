import ctypes
import multiprocessing as mp
import os
import sys
import traceback

import torch
import torch_npu  # noqa: F401

from ptodsl import do_bench
from ptodsl.test_util import get_test_device


M = 4224
N = 16384
K = 16384
BLOCK_DIM = 24
SWIZZLE_DIRECTION = 1
SWIZZLE_COUNT = 3
WARMUP_ITERS = 5
BENCH_ITERS = 20
TIMEOUT_SECONDS = 30.0


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.call_kernel.restype = None

    def matmul_abt(
        a,
        b,
        *,
        block_dim=BLOCK_DIM,
        swizzle_direction=SWIZZLE_DIRECTION,
        swizzle_count=SWIZZLE_COUNT,
        stream_ptr=None,
    ):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        m = int(a.shape[0])
        k = int(a.shape[1])
        n = int(b.shape[0])
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)

        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            torch_to_ctypes(c),
            m,
            n,
            k,
            swizzle_direction,
            swizzle_count,
        )
        return c

    return matmul_abt


def run_benchmark():
    device = get_test_device()
    torch.npu.set_device(device)
    matmul_abt = load_lib("./matmul_kernel.so")

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16, device=device)
    b = torch.randn(N, K, dtype=torch.float16, device=device)

    fn = lambda: matmul_abt(a, b)
    time_us = do_bench(
        fn,
        warmup_iters=WARMUP_ITERS,
        benchmark_iters=BENCH_ITERS,
        unit="us",
        flush_cache=False,
    )

    flops = 2.0 * M * N * K
    tflops = flops / time_us / 1e6

    return tflops, time_us


def _worker(result_q):
    try:
        tflops, time_us = run_benchmark()
        result_q.put(
            {
                "ok": True,
                "tflops": float(tflops),
                "time_us": float(time_us),
            }
        )
    except Exception:
        result_q.put({"ok": False, "error": traceback.format_exc()})


def main():
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_worker, args=(result_q,))
    proc.start()
    proc.join(TIMEOUT_SECONDS)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(
            f"ERROR: benchmark timed out after {TIMEOUT_SECONDS:.1f}s; "
            "terminated hung kernel process.",
            file=sys.stderr,
        )
        raise SystemExit(124)

    if result_q.empty():
        code = proc.exitcode
        print(f"ERROR: benchmark process exited without result (exit_code={code}).", file=sys.stderr)
        raise SystemExit(1)

    result = result_q.get()
    if not result["ok"]:
        print(result["error"], file=sys.stderr)
        raise SystemExit(1)

    print("---")
    print(f"(m, n, k)=({M}, {N}, {K})")
    print(f"TFLOPS: {result['tflops']:.1f}")
    print(f"execution_time: {result['time_us']:.5f} us")


if __name__ == "__main__":
    main()
