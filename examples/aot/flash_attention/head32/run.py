#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
#
# AOT runner for the multi-pipe FA builder. All .so variants must be built
# beforehand by `bash compile.sh` (which produces fa.so, fa_32.so, fa_64.so,
# fa_128.so by default). This script only loads and invokes them.
#
#   * Correctness check: the default 8k variant (fa.so).
#   * Benchmark: 8k / 16k / 32k / 64k variants by default. Override with
#     FA_BENCH_LENGTHS=8192,32768 (each length must be a multiple of S1_TILE).

import ctypes
import os
import sys
import math

import torch
import torch_npu  # noqa: F401

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "kernels"))
import fa_builder  # noqa: E402

from ptodsl import do_bench  # noqa: E402
from ptodsl.utils.npu_info import get_num_cube_cores, get_test_device  # noqa: E402

ARTIFACT_DIR = os.path.join(THIS_DIR, "build_artifacts")

# Sequence lengths benchmarked. Override with
#   FA_BENCH_LENGTHS=8192,32768
# Each must be a multiple of S1_TILE (=512) and have a matching prebuilt .so.
DEFAULT_BENCH_LENGTHS = (8192, 16384, 32768, 65536)


def _parse_bench_lengths():
    raw = os.environ.get("FA_BENCH_LENGTHS")
    if not raw:
        return DEFAULT_BENCH_LENGTHS
    return tuple(int(x) for x in raw.split(",") if x.strip())


ATOL = 1e-3
RTOL = 1e-3


def get_block_dim() -> int:
    return min(fa_builder.NUM_Q_BLOCKS, get_num_cube_cores())


def get_slot_elems(block_dim: int) -> int:
    return fa_builder.GM_ELEMS_PER_BLOCK * block_dim


def num_tiles_for(seq_len: int) -> int:
    s1_tile = fa_builder.S1_TILE
    if seq_len % s1_tile != 0:
        raise ValueError(f"seq_len {seq_len} not divisible by S1_TILE={s1_tile}")
    return seq_len // s1_tile


def lib_path_for(num_tiles: int) -> str:
    # NUM_TILES=16 is the builder default and produces plain fa.so.
    if num_tiles == 16:
        return os.path.join(ARTIFACT_DIR, "fa.so")
    return os.path.join(ARTIFACT_DIR, f"fa_{num_tiles}.so")


def require_lib(num_tiles: int) -> str:
    """Return the prebuilt .so path for the variant, or raise."""
    lib_path = lib_path_for(num_tiles)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Missing prebuilt kernel: {lib_path}\n"
            f"Run `bash compile.sh` (or `FA_TILES={num_tiles} bash compile.sh`) first."
        )
    return lib_path


def torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def load_lib(lib_path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(lib_path)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.call_kernel.restype = None
    return lib


def fa_reference(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = q.float() @ k.float().T * scale
    attn = torch.softmax(scores, dim=-1)
    return (attn @ v.float()).float()


def fused_attention(q, k, v, is_causal=False):
    scale = 1.0 / math.sqrt(q.shape[1])
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        num_heads=1,
        input_layout="BSH",
        scale=scale,
        next_tokens=0 if is_causal else 65535,
    )
    return out.squeeze(0)


def test_flash(lib, device):
    torch.manual_seed(0)
    Q_ROWS = fa_builder.Q_ROWS
    HEAD = fa_builder.HEAD
    S1_TOTAL = fa_builder.S1_TOTAL
    NUM_TILES = fa_builder.NUM_TILES
    GM_ELEMS_PER_BLOCK = fa_builder.GM_ELEMS_PER_BLOCK

    block_dim = get_block_dim()
    slot_elems = get_slot_elems(block_dim)

    q = torch.randn((Q_ROWS, HEAD), dtype=torch.float16, device=device)
    k = torch.randn((S1_TOTAL, HEAD), dtype=torch.float16, device=device)
    v = torch.randn((S1_TOTAL, HEAD), dtype=torch.float16, device=device)

    gm_slot = torch.zeros((slot_elems,), dtype=torch.float32, device=device)
    o = torch.zeros((Q_ROWS, HEAD), dtype=torch.float32, device=device)

    stream_ptr = torch.npu.current_stream()._as_parameter_

    lib.call_kernel(
        block_dim,
        stream_ptr,
        torch_to_ctypes(gm_slot),
        torch_to_ctypes(q),
        torch_to_ctypes(k),
        torch_to_ctypes(v),
        torch_to_ctypes(o),
    )
    torch.npu.synchronize()

    o_ref = fa_reference(q, k, v)
    torch.testing.assert_close(o.cpu().float(), o_ref.cpu(), rtol=RTOL, atol=ATOL)
    print(
        f"[fa] q_rows={Q_ROWS} s1={S1_TOTAL} head={HEAD} "
        f"({NUM_TILES} tiles, blockDim={block_dim}): PASSED "
        f"(atol={ATOL}, rtol={RTOL})  GM/blk={GM_ELEMS_PER_BLOCK} fp32"
    )


def benchmark_flash(lib, device, num_tiles, warmup=10, iters=100):
    """Benchmark a single (length-tagged) .so. Returns dict of metrics."""
    torch.manual_seed(0)
    Q_ROWS = fa_builder.Q_ROWS
    HEAD = fa_builder.HEAD
    S1_TILE = fa_builder.S1_TILE
    s1_total = S1_TILE * num_tiles

    block_dim = get_block_dim()
    slot_elems = get_slot_elems(block_dim)

    q = torch.randn((Q_ROWS, HEAD), dtype=torch.float16, device=device)
    k = torch.randn((s1_total, HEAD), dtype=torch.float16, device=device)
    v = torch.randn((s1_total, HEAD), dtype=torch.float16, device=device)

    gm_slot = torch.zeros((slot_elems,), dtype=torch.float32, device=device)
    o = torch.zeros((Q_ROWS, HEAD), dtype=torch.float32, device=device)
    stream_ptr = torch.npu.current_stream()._as_parameter_

    def run_kernel():
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(gm_slot),
            torch_to_ctypes(q),
            torch_to_ctypes(k),
            torch_to_ctypes(v),
            torch_to_ctypes(o),
        )

    def run_reference():
        fused_attention(q, k, v)

    kernel_us = do_bench(
        run_kernel, warmup_iters=warmup, benchmark_iters=iters, unit="us"
    )
    ref_us = do_bench(
        run_reference, warmup_iters=warmup, benchmark_iters=iters, unit="us"
    )

    # One untimed correctness check per length: assert against the fp32
    # reference so silent miscompiles fail loudly instead of just showing
    # a large max|err| in the summary table.
    run_kernel()
    torch.npu.synchronize()
    o_kernel = o.clone()
    o_fused = fused_attention(q, k, v)
    torch.npu.synchronize()
    o_golden = fa_reference(q, k, v)

    diff_kernel = (o_kernel.cpu().float() - o_golden.cpu()).abs().max().item()
    diff_fused = (o_fused.cpu().float() - o_golden.cpu()).abs().max().item()
    torch.testing.assert_close(
        o_kernel.cpu().float(), o_golden.cpu(), rtol=RTOL, atol=ATOL
    )

    flops = 4 * Q_ROWS * HEAD * s1_total
    return {
        "seq_len": s1_total,
        "num_tiles": num_tiles,
        "block_dim": block_dim,
        "kernel_us": kernel_us,
        "ref_us": ref_us,
        "kernel_gflops": flops / (kernel_us * 1e-6) / 1e9,
        "ref_gflops": flops / (ref_us * 1e-6) / 1e9,
        "speedup": ref_us / kernel_us,
        "kernel_max_err": diff_kernel,
        "fused_max_err": diff_fused,
    }


def print_bench_row(r):
    print(
        f"  s1={r['seq_len']:>6}  tiles={r['num_tiles']:>3}  "
        f"fa={r['kernel_us']:8.2f} us ({r['kernel_gflops']:7.1f} GF/s)  "
        f"ref={r['ref_us']:8.2f} us ({r['ref_gflops']:7.1f} GF/s)  "
        f"speedup={r['speedup']:.2f}x  "
        f"err: ours={r['kernel_max_err']:.2e} ref={r['fused_max_err']:.2e}"
    )


def main():
    device = get_test_device()
    torch.npu.set_device(device)

    bench_lengths = _parse_bench_lengths()

    # Verify all required .so artifacts exist before doing anything.
    required = [(seq_len, num_tiles_for(seq_len)) for seq_len in bench_lengths]
    for seq_len, nt in required:
        require_lib(nt)

    # ---- correctness on default 8k variant ----
    default_lib = load_lib(require_lib(16))
    test_flash(default_lib, device)

    # ---- benchmark across requested sequence lengths ----
    print(f"\n{'Benchmark (fa)':=^96}")
    print(
        f"  Q_ROWS={fa_builder.Q_ROWS}  HEAD={fa_builder.HEAD}  "
        f"S1_TILE={fa_builder.S1_TILE}  "
        f"NUM_Q_BLOCKS={fa_builder.NUM_Q_BLOCKS}  cores={get_num_cube_cores()}"
    )
    print(f"  lengths: {list(bench_lengths)}")
    print("-" * 96)

    results = []
    for seq_len, nt in required:
        lib = load_lib(require_lib(nt))
        r = benchmark_flash(lib, device, num_tiles=nt)
        print_bench_row(r)
        results.append(r)
    print("=" * 96)


if __name__ == "__main__":
    main()
