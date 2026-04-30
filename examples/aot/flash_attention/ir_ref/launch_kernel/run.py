#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
#
# Loads `build_artifacts/fa.so` built from `../fa.cpp` (ptoas-generated IR kernel).
# Host-side shapes must match the baked-in MLIR: Q_ROWS=2048, S1_TOTAL=4096,
# S1_TILE=256, NUM_TILES=16, HEAD=128 — kept in sync via `fa.build_env` + fa_performance_builder.
#
# Build first: `bash compile.sh`

import ctypes
import importlib
import math
import os
import sys

import torch
import torch_npu  # noqa: F401

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPLIT_PIPE_KERNELS = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "split_pipe", "kernels"))
sys.path.insert(0, _SPLIT_PIPE_KERNELS)
import fa_performance_builder as fb  # noqa: E402

from ptodsl import do_bench  # noqa: E402
from ptodsl.utils.npu_info import get_num_cube_cores, get_test_device  # noqa: E402

ARTIFACT_DIR = os.path.join(THIS_DIR, "build_artifacts")
DEFAULT_PLOT_PATH = os.path.join(ARTIFACT_DIR, "fa_benchmark.png")

# Single variant for bundled ../fa.cpp (4096 = 16 × 256).
DEFAULT_BENCH_LENGTHS = (4096,)


def _parse_bench_lengths():
    raw = os.environ.get("FA_BENCH_LENGTHS")
    if not raw:
        return DEFAULT_BENCH_LENGTHS
    return tuple(int(x) for x in raw.split(",") if x.strip())


ATOL = 1e-3
RTOL = 1e-3


def attn_flops_matmul_softmax_scale(
    batch_size: int,
    s_q: int,
    s_k: int,
    h: int,
    include_scale: bool = True,
    count_exp_as_flop: bool = True,
    count_max_as_flop: bool = True,
):
    flops_matmul = 4 * batch_size * s_q * s_k * h
    flops_scale = (batch_size * s_q * s_k) if include_scale else 0

    rows = batch_size * s_q
    softmax_ops = 0
    if count_max_as_flop:
        softmax_ops += rows * (s_k - 1)
    softmax_ops += rows * s_k
    if count_exp_as_flop:
        softmax_ops += rows * s_k
    softmax_ops += rows * (s_k - 1)
    softmax_ops += rows * s_k

    return flops_matmul + flops_scale + softmax_ops


def get_block_dim() -> int:
    # fa.cpp partitions NUM_TILES across get_block_num() / get_block_idx() (see cube/vec
    # loops), not NUM_Q_BLOCKS — using NUM_Q_BLOCKS mis-launches the grid and faults (507015).
    return min(fb.NUM_TILES, get_num_cube_cores())


def get_slot_elems(block_dim: int) -> int:
    return fb.GM_ELEMS_PER_BLOCK * block_dim


def num_tiles_for(seq_len: int) -> int:
    s1_tile = fb.S1_TILE
    if seq_len % s1_tile != 0:
        raise ValueError(f"seq_len {seq_len} not divisible by S1_TILE={s1_tile}")
    return seq_len // s1_tile


def _apply_build_env_matching_seq(seq_len: int) -> None:
    if not os.path.isdir(ARTIFACT_DIR):
        return
    matches = []
    for fn in sorted(os.listdir(ARTIFACT_DIR)):
        if not fn.endswith(".build_env"):
            continue
        path = os.path.join(ARTIFACT_DIR, fn)
        kv = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    k, _, v = line.partition("=")
                    if k.strip():
                        kv[k.strip()] = v.strip()
        except OSError:
            continue
        try:
            nt = int(kv["FA_NUM_TILES"])
            s1 = int(kv["FA_S1_TILE"])
        except (KeyError, ValueError):
            continue
        if nt * s1 != seq_len:
            continue
        matches.append((fn, kv))

    if not matches:
        return

    env_s1 = os.environ.get("FA_S1_TILE")
    if env_s1 is not None:
        for fn, kv in matches:
            if kv.get("FA_S1_TILE") == env_s1.strip():
                for k, v in kv.items():
                    os.environ[k] = v
                return

    matches.sort(key=lambda x: (0 if x[0] == "fa.build_env" else 1, x[0]))
    for _, kv in matches:
        for k, v in kv.items():
            os.environ[k] = v
        return


def lib_path_for(num_tiles: int) -> str:
    if num_tiles == 16:
        return os.path.join(ARTIFACT_DIR, "fa.so")
    return os.path.join(ARTIFACT_DIR, f"fa_{num_tiles}.so")


def require_lib(num_tiles: int) -> str:
    lib_path = lib_path_for(num_tiles)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Missing prebuilt kernel: {lib_path}\n"
            "Run `bash compile.sh` in ir_ref/launch_kernel first."
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


def test_flash(lib, device, num_tiles):
    torch.manual_seed(0)
    Q_ROWS = fb.Q_ROWS
    HEAD = fb.HEAD
    S1_TOTAL = fb.S1_TILE * num_tiles
    GM_ELEMS_PER_BLOCK = fb.GM_ELEMS_PER_BLOCK

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
        f"({num_tiles} tiles, blockDim={block_dim}): PASSED "
        f"(atol={ATOL}, rtol={RTOL})  GM/blk={GM_ELEMS_PER_BLOCK} fp32"
    )


def benchmark_flash(lib, device, num_tiles, warmup=10, iters=100):
    torch.manual_seed(0)
    Q_ROWS = fb.Q_ROWS
    HEAD = fb.HEAD
    S1_TILE = fb.S1_TILE
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
        run_kernel,
        warmup_iters=warmup,
        benchmark_iters=iters,
        unit="us",
        flush_cache=False,
    )
    ref_us = do_bench(
        run_reference,
        warmup_iters=warmup,
        benchmark_iters=iters,
        unit="us",
        flush_cache=False,
    )

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

    flops = attn_flops_matmul_softmax_scale(1, Q_ROWS, s1_total, HEAD)
    return {
        "seq_len": s1_total,
        "num_tiles": num_tiles,
        "block_dim": block_dim,
        "kernel_us": kernel_us,
        "ref_us": ref_us,
        "kernel_tflops": flops / (kernel_us * 1e-6) / 1e12,
        "ref_tflops": flops / (ref_us * 1e-6) / 1e12,
        "speedup": ref_us / kernel_us,
        "kernel_max_err": diff_kernel,
        "fused_max_err": diff_fused,
    }


def print_bench_row(r):
    print(
        f"  s1={r['seq_len']:>6}  tiles={r['num_tiles']:>3}  "
        f"fa={r['kernel_us']:8.2f} us ({r['kernel_tflops']:7.3f} TFLOP/s)  "
        f"ref={r['ref_us']:8.2f} us ({r['ref_tflops']:7.3f} TFLOP/s)  "
        f"speedup={r['speedup']:.2f}x  "
        f"err: ours={r['kernel_max_err']:.2e} ref={r['fused_max_err']:.2e}"
    )


def plot_benchmark_results(results, out_png=None):
    if not results:
        return

    out_png = out_png or os.environ.get("FA_BENCH_PLOT_PATH", DEFAULT_PLOT_PATH)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is not installed; skipping plot generation.")
        return

    style_candidates = ("seaborn-v0_8-whitegrid", "seaborn-whitegrid")
    for style_name in style_candidates:
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue

    seq_lens = [r["seq_len"] for r in results]
    fa_tflops = [r["kernel_tflops"] for r in results]
    ref_tflops = [r["ref_tflops"] for r in results]

    fig, ax_thr = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    ax_thr.plot(seq_lens, fa_tflops, "o-", label="PTO flash attention (ir_ref fa.cpp)")
    ax_thr.plot(seq_lens, ref_tflops, "s-", label="torch_npu fused attention")
    ax_thr.set_title("Throughput")
    ax_thr.set_xlabel("S1 sequence length")
    ax_thr.set_ylabel("TFLOP/s")
    ax_thr.legend()
    ax_thr.set_xscale("log", base=2)
    ax_thr.set_xticks(seq_lens)
    ax_thr.set_xticklabels([str(x) for x in seq_lens], rotation=30)

    fig.suptitle(
        f"Flash Attention Benchmark: Q={fb.Q_ROWS}, H={fb.HEAD}, "
        f"S1_TILE={fb.S1_TILE}"
    )
    fig.tight_layout()

    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved benchmark plot: {out_png}")


def main():
    global fb
    device = get_test_device()
    torch.npu.set_device(device)

    bench_lengths = _parse_bench_lengths()
    _apply_build_env_matching_seq(bench_lengths[0])
    fb = importlib.reload(fb)

    required = [(seq_len, num_tiles_for(seq_len)) for seq_len in bench_lengths]
    for seq_len, nt in required:
        require_lib(nt)

    _, first_nt = required[0]
    default_lib = load_lib(require_lib(first_nt))
    test_flash(default_lib, device, num_tiles=first_nt)

    print(f"\n{'Benchmark (fa)':=^96}")
    print(
        f"  Q_ROWS={fb.Q_ROWS}  HEAD={fb.HEAD}  "
        f"S1_TILE={fb.S1_TILE}  "
        f"NUM_Q_BLOCKS={fb.NUM_Q_BLOCKS}  cores={get_num_cube_cores()}"
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

    if os.environ.get("FA_BENCH_NO_PLOT", "").lower() not in ("1", "true", "yes"):
        plot_benchmark_results(results)


if __name__ == "__main__":
    main()
