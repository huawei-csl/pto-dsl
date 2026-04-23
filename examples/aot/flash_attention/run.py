#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
#
# Runner for the multi-pipe FA builder. Differences vs run.py:
#   * Uses the multi-pipe .so (built by compile.sh).
#   * Pulls per-block GM size from kernels/fa_builder.GM_ELEMS_PER_BLOCK
#     (3 separate pipes → 425 984 B/block instead of 262 144 B/block).

import ctypes
import os
import subprocess
import sys

import torch
import torch_npu  # noqa: F401

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, "kernels"))
from fa_builder import (  # noqa: E402
    GM_ELEMS_PER_BLOCK,
    HEAD,
    NUM_Q_BLOCKS,
    NUM_TILES,
    Q_ROWS,
    S0,
    S1_TILE,
    S1_TOTAL,
)

from ptodsl import do_bench  # noqa: E402
from ptodsl.utils.npu_info import get_num_cube_cores, get_test_device  # noqa: E402

DEFAULT_LIB_PATH = os.path.join(THIS_DIR, "build_artifacts", "fa.so")
DEFAULT_COMPILE_SCRIPT = os.path.join(THIS_DIR, "compile.sh")

ATOL = 1e-3
RTOL = 1e-3


def get_block_dim() -> int:
    return min(NUM_Q_BLOCKS, get_num_cube_cores())


def get_slot_elems(block_dim: int) -> int:
    return GM_ELEMS_PER_BLOCK * block_dim


def torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def compile_example(compile_script: str) -> None:
    subprocess.run(["bash", compile_script], check=True, cwd=THIS_DIR)


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
    import math

    scale = 1.0 / math.sqrt(q.shape[1])
    scores = q.float() @ k.float().T * scale
    attn = torch.softmax(scores, dim=-1)
    return (attn @ v.float()).float()


def fused_attention(q, k, v, is_causal=False):
    import math

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


def benchmark_flash(lib, device, warmup=10, iters=100):
    torch.manual_seed(0)
    block_dim = get_block_dim()
    slot_elems = get_slot_elems(block_dim)

    q = torch.randn((Q_ROWS, HEAD), dtype=torch.float16, device=device)
    k = torch.randn((S1_TOTAL, HEAD), dtype=torch.float16, device=device)
    v = torch.randn((S1_TOTAL, HEAD), dtype=torch.float16, device=device)

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

    run_kernel()
    torch.npu.synchronize()
    o_kernel = o.clone()
    o_fused = fused_attention(q, k, v)
    torch.npu.synchronize()
    o_golden = fa_reference(q, k, v)

    diff_kernel = (o_kernel.cpu().float() - o_golden.cpu()).abs()
    diff_fused = (o_fused.cpu().float() - o_golden.cpu()).abs()

    torch.testing.assert_close(
        o_kernel.cpu().float(), o_golden.cpu(), rtol=RTOL, atol=ATOL
    )

    flops = 4 * Q_ROWS * HEAD * S1_TOTAL
    print(f"\n{'Benchmark (fa)':=^60}")
    print(f"  q_rows={Q_ROWS}  s1={S1_TOTAL}  head={HEAD}  tiles={NUM_TILES}")
    print(
        f"  blockDim={block_dim}  Q-blocks={NUM_Q_BLOCKS}  cores={get_num_cube_cores()}"
    )
    print(f"  warmup={warmup}  iters={iters}")
    print(
        f"  fa:         {kernel_us:8.2f} us  ({flops / (kernel_us * 1e-6) / 1e9:.2f} GFLOP/s)"
    )
    print(
        f"  npu_fused_attn: {ref_us:8.2f} us  ({flops / ( ref_us * 1e-6) / 1e9:.2f} GFLOP/s)"
    )
    print(f"  speedup vs npu_fused_attn: {ref_us / kernel_us:.2f}x")
    print(
        f"  accuracy: kernel max|err|={diff_kernel.max():.2e}  "
        f"npu_fused max|err|={diff_fused.max():.2e}"
    )
    print(f"{'':=^60}")


def main():
    compile_example(DEFAULT_COMPILE_SCRIPT)
    device = get_test_device()
    torch.npu.set_device(device)
    lib = load_lib(DEFAULT_LIB_PATH)
    test_flash(lib, device)
    benchmark_flash(lib, device)


if __name__ == "__main__":
    main()
