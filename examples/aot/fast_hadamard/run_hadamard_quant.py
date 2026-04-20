import os
import argparse
import ctypes
import math

import torch
import torch_npu  # noqa: F401

from ptodsl.npu_info import get_num_cube_cores, get_test_device

ELEMENTS_PER_TILE = 32 * 1024 // 2  # 32KB UB / sizeof(fp16)
_DEFAULT_NUM_CORES = get_num_cube_cores()


# TODO: Fast hadamard quant breaks at pto-isa commit f2454146


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib_quant(lib_path, block_dim=_DEFAULT_NUM_CORES):
    """Load a fast_hadamard_quant shared library (fp16 → int8).

    The shared library must export::

        void call_fused_kernel(
            uint32_t blockDim, void *stream,
            uint8_t *x, uint8_t *y,
            uint8_t *group_scales, uint8_t *group_offsets,
            uint32_t scale_group_stride, uint32_t offset_group_stride,
            uint32_t batch, uint32_t n, uint32_t log2_n,
            float scale, uint32_t group_size, float q_offset);
    """
    lib = ctypes.CDLL(lib_path)
    lib.call_fused_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x      (fp16 input)
        ctypes.c_void_p,  # y      (int8 output)
        ctypes.c_void_p,  # group_scales  (fp16, or None)
        ctypes.c_void_p,  # group_offsets (fp16, or None)
        ctypes.c_uint32,  # scale_group_stride
        ctypes.c_uint32,  # offset_group_stride
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # n
        ctypes.c_uint32,  # log2_n
        ctypes.c_float,  # scale
        ctypes.c_uint32,  # group_size
        ctypes.c_float,  # q_offset
    ]
    lib.call_fused_kernel.restype = None

    def hadamard_quant_func(
        x,
        y,
        batch,
        n,
        log2_n,
        scale=1.0,
        q_offset=0.0,
        group_scales=None,
        group_offsets=None,
        group_size=0,
        scale_group_stride=0,
        offset_group_stride=0,
        block_dim=block_dim,
        stream_ptr=None,
    ):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        if n > ELEMENTS_PER_TILE or n <= 0 or (n & (n - 1)) != 0:
            raise ValueError(
                f"n must be a power of two in [1, {ELEMENTS_PER_TILE}], got {n}"
            )
        expected_log2_n = int(math.log2(n))
        if log2_n != expected_log2_n:
            raise ValueError(
                f"log2_n must equal log2(n)={expected_log2_n}, got {log2_n}"
            )
        lib.call_fused_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(group_scales) if group_scales is not None else None,
            torch_to_ctypes(group_offsets) if group_offsets is not None else None,
            scale_group_stride,
            offset_group_stride,
            batch,
            n,
            log2_n,
            scale,
            group_size,
            q_offset,
        )

    return hadamard_quant_func


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


def hadamard_quant_ref(
    x, scale=1.0, q_offset=0.0, group_scales=None, group_offsets=None, group_size=None
):
    """Reference fused Hadamard + quantize (fp16 → int8).

    Mirrors the kernel's arithmetic exactly:
    - Hadamard runs in fp16.
    - Group scales/offsets are applied in fp16 (same as tile.muls/tile.adds on the NPU).
    - Uniform scale/q_offset are applied as fp16 scalars.
    - Final clamp+cast to int8 uses truncation, matching tile.cvt rmode="none".

    group_scales / group_offsets must be fp16 tensors of shape
    [groups_per_row] (shared across rows) or [batch, groups_per_row] (per-row).
    group_size must match the value the kernel library was compiled with.
    """
    x = hadamard_ref_inplace(x)  # stays fp16

    if group_scales is not None or group_offsets is not None:
        assert group_size is not None, "group_size required with per-group params"
        n = x.shape[-1]
        groups_per_row = n // group_size
        x_grouped = x.reshape(*x.shape[:-1], groups_per_row, group_size)
        if group_scales is not None:
            gs = group_scales
            if gs.dim() == 1:
                gs = gs.unsqueeze(0)
            x_grouped = x_grouped * gs.unsqueeze(-1)
        else:
            x_grouped = x_grouped * torch.tensor(scale, dtype=x.dtype)
        if group_offsets is not None:
            go = group_offsets
            if go.dim() == 1:
                go = go.unsqueeze(0)
            x_grouped = x_grouped + go.unsqueeze(-1)
        else:
            x_grouped = x_grouped + torch.tensor(q_offset, dtype=x.dtype)
        x = x_grouped.reshape_as(x)
    else:
        x = x * torch.tensor(scale, dtype=x.dtype) + torch.tensor(
            q_offset, dtype=x.dtype
        )

    return x.float().to(torch.int32).to(torch.int8)


def test_hadamard_quant(quant_func, block_dim=_DEFAULT_NUM_CORES):
    """Correctness test for the fused Hadamard+quantize kernel.

    Coverage mirrors the pytest suite:
    - seeds: [0, 1]
    - scales: [0.5, 1.0, 2.0]
    - batches: [1, 7, 65]
    - n: [128, 1024, 16384]
    - mutation check: input x must not be modified
    - validation: non-power-of-two n and wrong log2_n raise ValueError
    """
    batch_list = [1, 7, 65]
    n_list = [128, 1024, 16384]
    scale_list = [0.5, 1.0, 2.0]
    seed_list = [0, 1]

    results = []
    for seed in seed_list:
        for scale in scale_list:
            for batch in batch_list:
                for n in n_list:
                    log2_n = int(math.log2(n))
                    torch.manual_seed(seed)
                    x = torch.randn(batch, n, device=device, dtype=torch.float16)
                    y = torch.zeros(batch, n, device=device, dtype=torch.int8)
                    y_ref = hadamard_quant_ref(x.cpu(), scale=scale).to(device)

                    quant_func(x, y, batch, n, log2_n, scale=scale, block_dim=block_dim)
                    torch.npu.synchronize()

                    is_match = True
                    detail = ""
                    try:
                        torch.testing.assert_close(y, y_ref)
                    except AssertionError as err:
                        is_match = False
                        detail = str(err).strip() if str(err) else "assert_close failed"

                    status = "match" if is_match else "mismatch"
                    print(
                        f"[{status}] seed={seed}, scale={scale}, batch={batch}, n={n}"
                    )
                    if detail:
                        print(f"  detail: {detail}")
                    results.append((seed, scale, batch, n, status, detail))

    # Mutation check: input must not be modified by the kernel.
    torch.manual_seed(0)
    x_mut = torch.randn(7, 1024, device=device, dtype=torch.float16)
    x_before = x_mut.clone()
    y_mut = torch.zeros(7, 1024, device=device, dtype=torch.int8)
    quant_func(x_mut, y_mut, 7, 1024, 10, scale=1.0, block_dim=block_dim)
    torch.npu.synchronize()
    if torch.equal(x_mut, x_before):
        print("[match] mutation check: input unchanged")
    else:
        print("[mismatch] mutation check: input was modified")
        results.append((None, None, 7, 1024, "mismatch", "input mutated"))

    # Validation: non-power-of-two n must raise ValueError.
    for bad_n in [3, 257, 16385]:
        try:
            quant_func(
                torch.zeros(2, bad_n, device=device, dtype=torch.float16),
                torch.zeros(2, bad_n, device=device, dtype=torch.int8),
                2,
                bad_n,
                int(math.log2(bad_n)) if bad_n > 0 else 0,
                scale=1.0,
            )
            print(f"[mismatch] expected ValueError for n={bad_n}")
            results.append((None, None, 2, bad_n, "mismatch", "expected ValueError"))
        except ValueError:
            print(f"[match] ValueError raised for n={bad_n}")

    # Validation: wrong log2_n must raise ValueError.
    try:
        quant_func(
            torch.zeros(2, 1024, device=device, dtype=torch.float16),
            torch.zeros(2, 1024, device=device, dtype=torch.int8),
            2,
            1024,
            9,
            scale=1.0,
        )
        print("[mismatch] expected ValueError for wrong log2_n")
        results.append(
            (None, None, 2, 1024, "mismatch", "expected ValueError for log2_n")
        )
    except ValueError:
        print("[match] ValueError raised for wrong log2_n")

    print("\nsummary:")
    for seed, scale, batch, n, status, detail in results:
        print(f"  seed={seed}, scale={scale}, batch={batch}, n={n}, status={status}")
    return results


def test_hadamard_quant_grouped(
    quant_func, group_size=128, block_dim=_DEFAULT_NUM_CORES
):
    """Correctness test for the per-group quantization path.

    Tests all combinations of shared (1-D) and per-row (2-D) group_scales /
    group_offsets.  group_size must match the value the library was compiled with.
    """
    results = []

    def _run(
        desc,
        batch,
        n,
        gs,
        group_scales,
        group_offsets,
        scale_group_stride,
        offset_group_stride,
    ):
        log2_n = int(math.log2(n))
        torch.manual_seed(0)
        x = torch.randn(batch, n, device=device, dtype=torch.float16)
        y = torch.zeros(batch, n, device=device, dtype=torch.int8)

        y_ref = hadamard_quant_ref(
            x.cpu(),
            scale=1.0,
            q_offset=0.0,
            group_scales=group_scales.cpu() if group_scales is not None else None,
            group_offsets=group_offsets.cpu() if group_offsets is not None else None,
            group_size=gs,
        ).to(device)

        torch.npu.synchronize()

        quant_func(
            x,
            y,
            batch,
            n,
            log2_n,
            scale=1.0,
            q_offset=0.0,
            group_scales=group_scales,
            group_offsets=group_offsets,
            group_size=gs,
            scale_group_stride=scale_group_stride,
            offset_group_stride=offset_group_stride,
            block_dim=block_dim,
        )
        torch.npu.synchronize()

        is_match = True
        detail = ""
        try:
            torch.testing.assert_close(y, y_ref)
        except AssertionError as err:
            is_match = False
            detail = str(err).strip() if str(err) else "assert_close failed"

        status = "match" if is_match else "mismatch"
        print(f"[{status}] {desc}, batch={batch}, n={n}, gs={gs}")
        if detail:
            print(f"  detail: {detail}")
        results.append((desc, batch, n, gs, status, detail))

    # shared 1-D scales, no offsets
    batch, n, gs = 5, 1024, group_size
    groups = n // gs
    _run(
        "shared_scales",
        batch,
        n,
        gs,
        group_scales=torch.linspace(
            0.5, 1.5, groups, device=device, dtype=torch.float16
        ).contiguous(),
        group_offsets=None,
        scale_group_stride=0,
        offset_group_stride=0,
    )

    # shared 1-D scales + offsets
    batch, n, gs = 3, 1024, group_size
    groups = n // gs
    _run(
        "shared_scales_and_offsets",
        batch,
        n,
        gs,
        group_scales=torch.linspace(
            0.75, 1.25, groups, device=device, dtype=torch.float16
        ).contiguous(),
        group_offsets=torch.linspace(
            -1.0, 1.0, groups, device=device, dtype=torch.float16
        ).contiguous(),
        scale_group_stride=0,
        offset_group_stride=0,
    )

    # per-row 2-D scales, no offsets
    batch, n, gs = 5, 1024, group_size
    groups = n // gs
    _run(
        "per_row_scales",
        batch,
        n,
        gs,
        group_scales=torch.stack(
            [
                torch.linspace(
                    0.5 + 0.1 * i, 1.5 + 0.1 * i, groups, dtype=torch.float16
                )
                for i in range(batch)
            ]
        )
        .to(device)
        .contiguous(),
        group_offsets=None,
        scale_group_stride=groups,
        offset_group_stride=0,
    )

    # per-row 2-D scales + offsets
    batch, n, gs = 7, 1024, group_size
    groups = n // gs
    _run(
        "per_row_scales_and_offsets",
        batch,
        n,
        gs,
        group_scales=torch.stack(
            [
                torch.linspace(
                    0.75 + 0.05 * i, 1.25 + 0.05 * i, groups, dtype=torch.float16
                )
                for i in range(batch)
            ]
        )
        .to(device)
        .contiguous(),
        group_offsets=torch.stack(
            [
                torch.linspace(
                    -1.0 + 0.1 * i, 1.0 + 0.1 * i, groups, dtype=torch.float16
                )
                for i in range(batch)
            ]
        )
        .to(device)
        .contiguous(),
        scale_group_stride=groups,
        offset_group_stride=groups,
    )

    print("\nsummary:")
    for desc, batch, n, gs, status, detail in results:
        print(f"  {desc}, batch={batch}, n={n}, gs={gs}, status={status}")
    return results


def benchmark_quant(
    quant_func,
    scale=1.0 / 128.0,
    q_offset=0.0,
    warmup=2,
    repeats=20,
    output_dir="./perf_data/",
    manual_sync=False,
):
    """Benchmark the fused Hadamard+quantize kernel across (batch, N, block_dim) configs."""
    TEST_HIDDEN_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    BENCH_BATCHES = [1, 5, 8, 10, 16, 20, 32, 40, 64, 128, 256, 512, 1024]
    BENCH_BLOCK_DIMS = [20, _DEFAULT_NUM_CORES]

    os.makedirs(output_dir, exist_ok=True)

    for block_dim in BENCH_BLOCK_DIMS:
        print(f"\n{'=' * 60}")
        print(f"BENCHMARK quant (BLOCK_DIM={block_dim})")
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

                x_list = [
                    torch.randn(batch, n, device="npu", dtype=torch.float16)
                    for _ in range(allocated)
                ]
                y_list = [
                    torch.zeros(batch, n, device="npu", dtype=torch.int8)
                    for _ in range(allocated)
                ]

                for i in range(warmup):
                    quant_func(
                        x_list[i],
                        y_list[i],
                        batch,
                        n,
                        log2_n,
                        scale=scale,
                        q_offset=q_offset,
                        block_dim=block_dim,
                    )
                torch.npu.synchronize()

                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)

                start.record()
                for i in range(repeats):
                    quant_func(
                        x_list[warmup + i],
                        y_list[warmup + i],
                        batch,
                        n,
                        log2_n,
                        scale=scale,
                        q_offset=q_offset,
                        block_dim=block_dim,
                    )
                end.record()
                torch.npu.synchronize()

                duration_ms = start.elapsed_time(end) / repeats
                dur_us = duration_ms * 1e3

                # Read fp16 + write int8: batch * n * (2 + 1) bytes
                data_bytes = batch * n * 3
                bw_gbs = (data_bytes / 1e9) / (dur_us / 1e6) if dur_us > 0 else 0.0

                print(f"{batch:>6d}  {n:>6d}  {dur_us:>12.2f}  {bw_gbs:>14.2f}")
                records.append(f"{batch},{n},{dur_us:.4f},{bw_gbs:.4f}")

        suffix = "_manual" if manual_sync else ""
        csv_path = os.path.join(output_dir, f"fht_quant_pto{suffix}_bd{block_dim}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("batch,N,duration_us,bandwidth_gbs\n")
            f.write("\n".join(records) + "\n")
        print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test and benchmark the fused Hadamard+quantize (fp16 → int8) kernel."
    )
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Use manual-sync library instead of the default auto-sync library.",
    )
    parser.add_argument(
        "--block-dim",
        type=int,
        default=_DEFAULT_NUM_CORES,
        help=f"Kernel blockDim (default: {_DEFAULT_NUM_CORES}).",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size used when building the per-group quant library (default: 128).",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip the benchmark loop (correctness tests only).",
    )
    args = parser.parse_args()

    device = get_test_device()
    torch.npu.set_device(device)

    lib_path = (
        "./hadamard_quant_manual_sync_lib.so"
        if args.manual_sync
        else "./hadamard_quant_lib.so"
    )
    quant_func = load_lib_quant(lib_path=lib_path, block_dim=args.block_dim)
    test_hadamard_quant(quant_func, block_dim=args.block_dim)

    gs = args.group_size
    gs_lib_path = (
        f"./hadamard_quant_gs{gs}_manual_sync_lib.so"
        if args.manual_sync
        else f"./hadamard_quant_gs{gs}_lib.so"
    )
    quant_func_gs = load_lib_quant(lib_path=gs_lib_path, block_dim=args.block_dim)
    test_hadamard_quant_grouped(quant_func_gs, group_size=gs, block_dim=args.block_dim)

    if not args.no_benchmark:
        benchmark_quant(quant_func, manual_sync=args.manual_sync)
