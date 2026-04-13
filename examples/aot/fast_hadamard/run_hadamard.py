import os
import argparse
import ctypes
import csv
import math

import torch
import torch_npu  # noqa: F401

from ptodsl.npu_info import get_num_cube_cores, get_test_device

ELEMENTS_PER_TILE = 32 * 1024 // 2  # 32KB UB / sizeof(fp16)
_DEFAULT_NUM_CORES = get_num_cube_cores()


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, block_dim=_DEFAULT_NUM_CORES):
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x (in-place)
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # n
        ctypes.c_uint32,  # log2_n
    ]
    lib.call_kernel.restype = None

    def hadamard_func(x, batch, n, log2_n, block_dim=block_dim, stream_ptr=None):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        assert n <= ELEMENTS_PER_TILE, f"n must be <= {ELEMENTS_PER_TILE}, got {n}"
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            batch,
            n,
            log2_n,
        )

    return hadamard_func


def hadamard_ref_inplace(x):
    """Reference FHT matching TGATHER(P0101/P1010) + TADD/TSUB layout."""
    x = x.clone()
    n = x.shape[-1]
    n_half = n // 2
    log2_n = int(math.log2(n))
    for _ in range(log2_n):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd
    return x


def _is_power_of_two(v):
    return v > 0 and (v & (v - 1)) == 0


def test_hadamard(hadamard_func, block_dim=_DEFAULT_NUM_CORES):
    torch.manual_seed(0)
    dtype = torch.float16
    batch_list = [1, 7, 29, 65]
    n_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    results = []
    for batch in batch_list:
        for n in n_list:
            if not _is_power_of_two(n):
                continue
            log2_n = int(math.log2(n))
            x = torch.randn(batch, n, device=device, dtype=dtype)
            y_ref = hadamard_ref_inplace(x)

            hadamard_func(x, batch, n, log2_n)
            torch.npu.synchronize()

            is_match = True
            detail = ""
            try:
                torch.testing.assert_close(x, y_ref)
            except AssertionError as err:
                is_match = False
                detail = str(err).strip() if str(err) else "assert_close failed"

            status = "match" if is_match else "mismatch"
            print(f"[{status}] batch={batch}, n={n}, lib={lib_path}")
            if detail:
                print("  detail:")
                print(detail)
            results.append((batch, n, status, detail))

    print(f"detailed summary for {lib_path}:")
    for batch, n, status, detail in results:
        msg = f"  batch={batch}, n={n}, status={status}"
        print(msg)
        if detail:
            print("    detail:")
            print(detail)
    return results


def benchmark(hadamard_func, warmup=2, repeats=20, output_dir="./perf_data/"):
    """Benchmark across (batch, N, block_dim) configs.

    Uses separate input tensors per run to avoid L2 cache reuse,
    and a single timing-event pair averaged over all runs.
    """
    TEST_HIDDEN_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    BENCH_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]
    BENCH_BLOCK_DIMS = [20, _DEFAULT_NUM_CORES]

    os.makedirs(output_dir, exist_ok=True)

    for block_dim in BENCH_BLOCK_DIMS:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK (BLOCK_DIM={block_dim})")
        print(f"{'=' * 60}")
        header = (
            f"{'batch':>6s}  {'N':>6s}"
            f"  {'duration_us':>12s}  {'bandwidth_gbs':>14s}"
        )
        print(header)
        print("-" * len(header))

        records = []

        for batch in BENCH_BATCHES:
            for n in TEST_HIDDEN_DIMS:
                log2_n = int(math.log2(n))
                allocated = warmup + repeats

                # Separate GM tensors to avoid L2 cache reuse
                x_list = [
                    torch.randn(batch, n, device="npu", dtype=torch.float16)
                    for _ in range(allocated)
                ]

                # Warmup
                for i in range(warmup):
                    hadamard_func(x_list[i], batch, n, log2_n, block_dim=block_dim)
                torch.npu.synchronize()

                # Timed runs — single event pair, average over repeats
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)

                start.record()
                for i in range(repeats):
                    hadamard_func(
                        x_list[warmup + i],
                        batch,
                        n,
                        log2_n,
                        block_dim=block_dim,
                    )
                end.record()
                torch.npu.synchronize()

                duration_ms = start.elapsed_time(end) / repeats
                dur_us = duration_ms * 1e3

                # Bandwidth: read + write = 2 * batch * n * sizeof(half)
                data_bytes = 2 * batch * n * 2
                bw_gbs = (data_bytes / 1e9) / (dur_us / 1e6) if dur_us > 0 else 0.0

                print(f"{batch:>6d}  {n:>6d}" f"  {dur_us:>12.2f}  {bw_gbs:>14.2f}")
                records.append(f"{batch},{n},{dur_us:.4f},{bw_gbs:.4f}")

        csv_path = os.path.join(output_dir, f"fht_pto_bd{block_dim}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("batch,N,duration_us,bandwidth_gbs\n")
            f.write("\n".join(records) + "\n")
        print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Use manual-sync library instead of the default auto-sync library.",
    )
    parser.add_argument(
        "--block-dim",
        type=int,
        default=24,
        help="Kernel blockDim (default: 24).",
    )
    args = parser.parse_args()

    lib_path = (
        "./hadamard_manual_sync_lib.so"
        if args.manual_sync
        else "./hadamard_auto_sync_lib.so"
    )

    device = get_test_device()
    torch.npu.set_device(device)
    hadamard_func = load_lib(lib_path=lib_path, block_dim=args.block_dim)

    test_hadamard(hadamard_func)
    benchmark(hadamard_func)
