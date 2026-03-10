import argparse
import ctypes
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device


BLOCK_DIM = 24
M_LIST = [128 * i for i in range(1, 37, 4)]  # 128, ..., 4224
SHAPES_NK = [
    (4096, 4096),
    (8192, 8192),
    (16384, 16384),
]
N_WARMUP = 5
N_REPEAT = 20


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
    ]
    lib.call_kernel.restype = None

    def matmul_abt(a, b, *, block_dim=BLOCK_DIM, stream_ptr=None):
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
        )
        return c

    return matmul_abt


def _parse_int_list(raw):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("List cannot be empty.")
    return [int(p) for p in parts]


def _time_us(fn, a_list, b_list, warmup, repeat):
    for a, b in zip(a_list[:warmup], b_list[:warmup]):
        fn(a, b)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for a, b in zip(a_list[warmup : warmup + repeat], b_list[warmup : warmup + repeat]):
        fn(a, b)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeat


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark double-buffer (auto/manual) and single-buffer variants."
    )
    parser.add_argument(
        "--auto-lib",
        type=str,
        default="./simple_matmul_auto_sync_kernel.so",
        help="Path to auto-sync shared library.",
    )
    parser.add_argument(
        "--manual-lib",
        type=str,
        default="./simple_matmul_manual_sync_kernel.so",
        help="Path to manual-sync shared library.",
    )
    parser.add_argument(
        "--single-lib",
        type=str,
        default="./single_buffer_matmul_auto_sync_kernel.so",
        help="Path to single-buffer auto-sync shared library.",
    )
    parser.add_argument(
        "--m-list",
        type=str,
        default=",".join(str(m) for m in M_LIST),
        help="Comma-separated M values (default: script M_LIST).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=N_WARMUP,
        help=f"Warmup iterations (default: {N_WARMUP}).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=N_REPEAT,
        help=f"Timed iterations (default: {N_REPEAT}).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.warmup < 1 or args.repeat < 1:
        raise ValueError("--warmup and --repeat must be positive integers.")

    base_dir = Path(__file__).resolve().parent

    auto_lib = Path(args.auto_lib)
    if not auto_lib.is_absolute():
        auto_lib = base_dir / auto_lib
    manual_lib = Path(args.manual_lib)
    if not manual_lib.is_absolute():
        manual_lib = base_dir / manual_lib
    single_lib = Path(args.single_lib)
    if not single_lib.is_absolute():
        single_lib = base_dir / single_lib
    if not auto_lib.exists():
        raise FileNotFoundError(f"Auto-sync library not found: {auto_lib}")
    if not manual_lib.exists():
        raise FileNotFoundError(f"Manual-sync library not found: {manual_lib}")
    if not single_lib.exists():
        raise FileNotFoundError(f"Single-buffer library not found: {single_lib}")

    device = get_test_device()
    torch.npu.set_device(device)
    torch.manual_seed(0)

    auto_mm = load_lib(str(auto_lib))
    manual_mm = load_lib(str(manual_lib))
    single_mm = load_lib(str(single_lib))
    m_list = _parse_int_list(args.m_list)

    ratios_auto_vs_manual = []
    ratios_single_vs_auto = []
    ratios_single_vs_manual = []
    print(f"auto-sync lib:   {auto_lib}")
    print(f"manual-sync lib: {manual_lib}")
    print(f"single-buffer lib: {single_lib}")
    print("")

    for n, k in SHAPES_NK:
        print(f"=== N={n}, K={k} ===")
        for m in m_list:
            alloc = args.warmup + args.repeat
            a_list = [torch.randn(m, k, dtype=torch.float16, device=device) for _ in range(alloc)]
            b_list = [torch.randn(n, k, dtype=torch.float16, device=device) for _ in range(alloc)]

            auto_us = _time_us(auto_mm, a_list, b_list, args.warmup, args.repeat)
            manual_us = _time_us(manual_mm, a_list, b_list, args.warmup, args.repeat)
            single_us = _time_us(single_mm, a_list, b_list, args.warmup, args.repeat)

            flops = 2.0 * m * n * k
            auto_tflops = flops / auto_us / 1e6
            manual_tflops = flops / manual_us / 1e6
            single_tflops = flops / single_us / 1e6
            auto_vs_manual = manual_us / auto_us
            single_vs_auto = auto_us / single_us
            single_vs_manual = manual_us / single_us
            ratios_auto_vs_manual.append(auto_vs_manual)
            ratios_single_vs_auto.append(single_vs_auto)
            ratios_single_vs_manual.append(single_vs_manual)

            print(
                f"(M,N,K)=({m},{n},{k}) "
                f"auto={auto_tflops:.3f}TF, manual={manual_tflops:.3f}TF, single={single_tflops:.3f}TF, "
                f"ratio(auto/manual)={auto_vs_manual:.3f}x, "
                f"ratio(auto/single)={single_vs_auto:.3f}x, "
                f"ratio(manual/single)={single_vs_manual:.3f}x"
            )
        print("")

    avg_auto_vs_manual = sum(ratios_auto_vs_manual) / len(ratios_auto_vs_manual)
    min_auto_vs_manual = min(ratios_auto_vs_manual)
    max_auto_vs_manual = max(ratios_auto_vs_manual)
    avg_single_vs_auto = sum(ratios_single_vs_auto) / len(ratios_single_vs_auto)
    min_single_vs_auto = min(ratios_single_vs_auto)
    max_single_vs_auto = max(ratios_single_vs_auto)
    avg_single_vs_manual = sum(ratios_single_vs_manual) / len(ratios_single_vs_manual)
    min_single_vs_manual = min(ratios_single_vs_manual)
    max_single_vs_manual = max(ratios_single_vs_manual)

    print("=== Summary ===")
    print(f"avg ratio(auto/manual): {avg_auto_vs_manual:.3f}x")
    print(f"min ratio(auto/manual): {min_auto_vs_manual:.3f}x")
    print(f"max ratio(auto/manual): {max_auto_vs_manual:.3f}x")
    print(f"avg ratio(auto/single): {avg_single_vs_auto:.3f}x")
    print(f"min ratio(auto/single): {min_single_vs_auto:.3f}x")
    print(f"max ratio(auto/single): {max_single_vs_auto:.3f}x")
    print(f"avg ratio(manual/single): {avg_single_vs_manual:.3f}x")
    print(f"min ratio(manual/single): {min_single_vs_manual:.3f}x")
    print(f"max ratio(manual/single): {max_single_vs_manual:.3f}x")

    if avg_single_vs_auto >= 1.0:
        print(f"auto-sync double-buffer is faster than single-buffer by {(avg_single_vs_auto - 1.0) * 100.0:.2f}% on average")
    else:
        print(f"single-buffer is faster than auto-sync double-buffer by {(1.0 - avg_single_vs_auto) * 100.0:.2f}% on average")


if __name__ == "__main__":
    main()
