#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the
# terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance
# with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import warnings

warnings.filterwarnings(
    "ignore", message=".*owner does not match the current owner.*", category=UserWarning
)
import random
import math
import argparse
import ctypes
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch_npu
from ptodsl.utils import get_test_device
from ptodsl.bench import do_bench


THIS_DIR = Path(__file__).resolve().parent

_DEVICE = get_test_device()
torch.npu.set_device(_DEVICE)

NUM_ITERATIONS = 15
WARMUP = 10
SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
torch.npu.manual_seed(SEED)


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


def tflops(flops: int, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


# ---------------------------
# 2) Reference attention (pure PyTorch, fp32)
# ---------------------------
def fa_reference(q, k, v, is_causal=False):
    scale = 1.0 / math.sqrt(q.shape[1])
    scores = q.float() @ k.float().T * scale
    if is_causal:
        mask = torch.triu(
            torch.ones(scores.shape, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v.float()


def fused_attention(q, k, v, is_causal=False):
    scale = 1.0 / math.sqrt(q.shape[1])
    # npu_fused_infer_attention_score expects BSH: (1, S, H)
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


def torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def load_dsl_flash(lib_path: Path | None = None):
    if lib_path is None:
        lib_path = THIS_DIR / "build_artifacts" / "fa_dsl.so"
        print(f"Using default lib path: {lib_path}")
    builder_path = THIS_DIR / "build_artifacts" / "fa_dsl_runtime_builder.py"
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Missing {lib_path}; run compile.sh or compile_tile512.sh first"
        )
    if not builder_path.exists():
        raise FileNotFoundError(
            f"Missing {builder_path}; run compile.sh or compile_tile512.sh first"
        )

    spec = importlib.util.spec_from_file_location("fa_builder_runtime", builder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import builder from {builder_path}")
    fa_dsl_builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fa_dsl_builder)

    lib = ctypes.CDLL(str(lib_path))
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.call_kernel.restype = None

    ws = {}

    def alloc_workspace(s0: int, s1: int, head: int, device):
        shape = (s0, s1, head, str(device))
        if ws.get("shape") == shape:
            return
        block_dim = s0 // fa_dsl_builder.CUBE_S0
        ws.clear()
        ws["shape"] = shape
        ws["gm_slot"] = torch.empty(
            (fa_dsl_builder.GM_ELEMS_PER_BLOCK * block_dim,),
            dtype=torch.float32,
            device=device,
        )
        ws["o"] = torch.empty((s0, head), dtype=torch.float32, device=device)
        ws["block_dim"] = block_dim

    def flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.shape[1] != fa_dsl_builder.HEAD:
            raise ValueError(f"HEAD must be {fa_dsl_builder.HEAD}, got {q.shape[1]}")
        if q.shape[0] % fa_dsl_builder.CUBE_S0 != 0:
            raise ValueError(
                f"S0 must be divisible by CUBE_S0={fa_dsl_builder.CUBE_S0}"
            )
        if k.shape[0] % fa_dsl_builder.TILE_S1 != 0:
            raise ValueError(
                f"S1 must be divisible by TILE_S1={fa_dsl_builder.TILE_S1}"
            )

        alloc_workspace(q.shape[0], k.shape[0], q.shape[1], q.device)
        block_dim = q.shape[0] // fa_dsl_builder.CUBE_S0
        stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(ws["gm_slot"]),
            torch_to_ctypes(q),
            torch_to_ctypes(k),
            torch_to_ctypes(v),
            torch_to_ctypes(ws["o"]),
            q.shape[0],
            k.shape[0],
        )
        return ws["o"]

    def prepare_raw_runner(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.shape[1] != fa_dsl_builder.HEAD:
            raise ValueError(f"HEAD must be {fa_dsl_builder.HEAD}, got {q.shape[1]}")
        if q.shape[0] % fa_dsl_builder.CUBE_S0 != 0:
            raise ValueError(
                f"S0 must be divisible by CUBE_S0={fa_dsl_builder.CUBE_S0}"
            )
        if k.shape[0] % fa_dsl_builder.TILE_S1 != 0:
            raise ValueError(
                f"S1 must be divisible by TILE_S1={fa_dsl_builder.TILE_S1}"
            )

        alloc_workspace(q.shape[0], k.shape[0], q.shape[1], q.device)
        block_dim = ws["block_dim"]
        stream_ptr = torch.npu.current_stream()._as_parameter_
        gm_slot_ptr = torch_to_ctypes(ws["gm_slot"])
        q_ptr = torch_to_ctypes(q)
        k_ptr = torch_to_ctypes(k)
        v_ptr = torch_to_ctypes(v)
        o_ptr = torch_to_ctypes(ws["o"])
        q_rows = q.shape[0]
        k_rows = k.shape[0]

        def run_raw():
            lib.call_kernel(
                block_dim,
                stream_ptr,
                gm_slot_ptr,
                q_ptr,
                k_ptr,
                v_ptr,
                o_ptr,
                q_rows,
                k_rows,
            )
            return ws["o"]

        return run_raw

    return flash, prepare_raw_runner, fa_dsl_builder.TILE_S1, fa_dsl_builder.QK_PRELOAD


def parse_s1_values(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def test_flash(s1_values: list[int] | None = None, perf_mode: bool = False):
    s0, head = 128 * 24, 128
    default_s1_values = [1024, 2048, 4096, 8192, 16384, 32768, 64 * 1024, 128 * 1024]
    user_provided_s1 = s1_values is not None
    if s1_values is None:
        s1_values = default_s1_values
    is_causal = False

    dtype = torch.float16
    q2d = torch.randn((s0, head), dtype=dtype).npu()

    flash, prepare_raw_runner, s1_tile, qk_preload = load_dsl_flash()
    min_s1 = s1_tile * qk_preload
    invalid_s1 = [s1 for s1 in s1_values if s1 < min_s1]
    if invalid_s1 and user_provided_s1:
        raise ValueError(
            f"S1 values {invalid_s1} are too small for TILE_S1={s1_tile} "
            f"and QK_PRELOAD={qk_preload}; minimum S1 is {min_s1}"
        )
    if invalid_s1:
        print(
            f"Skipping S1 values {invalid_s1}: TILE_S1={s1_tile}, "
            f"QK_PRELOAD={qk_preload}, minimum S1={min_s1}"
        )
        s1_values = [s1 for s1 in s1_values if s1 >= min_s1]
    run_flash = lambda q, k, v: flash(q, k, v)

    flash_ms_values = []
    npu_ms_values = []
    ref_ms_values = []
    flash_tflops_values = []
    npu_tflops_values = []
    ref_tflops_values = []
    cases = [
        {"s1": s1, "flops_total": attn_flops_matmul_softmax_scale(1, s0, s1, head)}
        for s1 in s1_values
    ]

    def make_kv(s1: int):
        torch.manual_seed(SEED + s1)
        k2d = torch.randn((s1, head), dtype=dtype).npu()
        v2d = torch.randn((s1, head), dtype=dtype).npu()
        return k2d, v2d

    if perf_mode:
        if len(cases) != 1:
            raise ValueError(
                "--perf-mode requires exactly one S1 via --perf-mode S1 or --s1-values"
            )
        case = cases[0]
        s1 = case["s1"]
        flops_total = case["flops_total"]
        k2d, v2d = make_kv(s1)
        run_flash_raw = prepare_raw_runner(q2d, k2d, v2d)
        flash_ms = do_bench(
            run_flash_raw,
            warmup_iters=WARMUP,
            benchmark_iters=NUM_ITERATIONS,
            unit="ms",
        )
        print("==== PTODSL perf mode ====")
        print(f"S1                         : {s1}")
        print(f"Causal                     : {is_causal}")
        print(f"GFLOPs total                : {flops_total//10e9}")
        print(
            f"{'PTODSL flash kernel':<27}: {flash_ms:.3f} ms/iter  "
            f"({tflops(flops_total, flash_ms):.3f} TFLOP/s)"
        )
        return

    print("==== Reference phase ====")
    for case in cases:
        s1 = case["s1"]
        flops_total = case["flops_total"]
        k2d, v2d = make_kv(s1)

        ref_ms = do_bench(
            lambda: fa_reference(q2d, k2d, v2d, is_causal=is_causal),
            warmup_iters=WARMUP,
            benchmark_iters=NUM_ITERATIONS,
            unit="ms",
        )
        npu_ms = do_bench(
            lambda: fused_attention(q2d, k2d, v2d, is_causal=is_causal),
            warmup_iters=WARMUP,
            benchmark_iters=NUM_ITERATIONS,
            unit="ms",
        )
        o_ref = fa_reference(q2d, k2d, v2d, is_causal=is_causal).to(torch.float32)
        o_npu = fused_attention(q2d, k2d, v2d, is_causal=is_causal).to(torch.float32)

        case["ref_ms"] = ref_ms
        case["npu_ms"] = npu_ms
        case["o_ref"] = o_ref
        case["o_npu"] = o_npu

        ref_ms_values.append(ref_ms)
        npu_ms_values.append(npu_ms)
        ref_tflops_values.append(tflops(flops_total, ref_ms))
        npu_tflops_values.append(tflops(flops_total, npu_ms))

        print(f"S1                         : {s1}")
        print(f"Causal                     : {is_causal}")
        print(f"GFLOPs total                : {flops_total//10e9}")
        print(
            f"npu_fused_infer_attention  : {npu_ms:.3f} ms/iter  "
            f"({tflops(flops_total, npu_ms):.3f} TFLOP/s)"
        )
        print(
            f"torch reference            : {ref_ms:.3f} ms/iter  "
            f"({tflops(flops_total, ref_ms):.3f} TFLOP/s)"
        )
        print("")
        del k2d, v2d

    print("==== PTODSL kernel phase ====")
    for case in cases:
        s1 = case["s1"]
        flops_total = case["flops_total"]
        k2d, v2d = make_kv(s1)
        o_ref = case["o_ref"]
        o_npu = case["o_npu"]
        run_flash_raw = prepare_raw_runner(q2d, k2d, v2d)

        flash_ms = do_bench(
            run_flash_raw,
            warmup_iters=WARMUP,
            benchmark_iters=NUM_ITERATIONS,
            unit="ms",
        )
        flash_ms_values.append(flash_ms)
        flash_tflops_values.append(tflops(flops_total, flash_ms))

        o_out = run_flash(q2d, k2d, v2d)

        print(f"S1                         : {s1}")
        print(f"Causal                     : {is_causal}")
        print(f"GFLOPs total                : {flops_total//10e9}")
        print(
            f"{'PTODSL flash kernel':<27}: {flash_ms:.3f} ms/iter  "
            f"({tflops(flops_total, flash_ms):.3f} TFLOP/s)"
        )
        print(
            f"npu_fused_infer_attention  : {case['npu_ms']:.3f} ms/iter  "
            f"({tflops(flops_total, case['npu_ms']):.3f} TFLOP/s)"
        )
        print(
            f"torch reference            : {case['ref_ms']:.3f} ms/iter  "
            f"({tflops(flops_total, case['ref_ms']):.3f} TFLOP/s)"
        )
        torch.testing.assert_close(o_out, o_ref, rtol=1e-3, atol=1e-3)
        print("vs torch reference: PASSED")
        torch.testing.assert_close(o_out, o_npu, rtol=1e-3, atol=1e-3)
        print("vs npu_fused_attention: PASSED")
        print("")
        del k2d, v2d, o_out

    plot_path = Path(__file__).with_name("naive_tpush_dsl_plot.png")
    plt.figure(figsize=(8, 5))
    plt.plot(s1_values, flash_tflops_values, marker="o", label="flash")
    plt.plot(s1_values, ref_tflops_values, marker="o", label="ref")
    plt.plot(s1_values, npu_tflops_values, marker="o", label="torch_npu")
    plt.xscale("log", base=2)
    plt.xticks(s1_values, [str(v) for v in s1_values])
    plt.xlabel("S1")
    plt.ylabel("TFLOP/s")
    plt.title(
        f"Flash Attention ptodsl vs rest. TFLOP/s vs S1\n(S0={s0}, head={head} S1_TILE={s1_tile})"
    )
    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s1-values",
        default=None,
        help="Comma-separated S1 values. Default: full sweep.",
    )
    parser.add_argument(
        "--perf-mode",
        nargs="?",
        const=True,
        default=False,
        metavar="S1",
        help=(
            "Benchmark only the PTODSL raw kernel. Optionally pass one S1 directly, "
            "for example: --perf-mode 131072. The old form with --s1-values still works."
        ),
    )
    args = parser.parse_args()
    perf_mode = args.perf_mode is not False
    s1_values = parse_s1_values(args.s1_values) if args.s1_values else None
    if args.perf_mode is not False and args.perf_mode is not True:
        if s1_values is not None:
            raise ValueError(
                "Pass S1 either as --perf-mode S1 or --s1-values, not both"
            )
        s1_values = [int(args.perf_mode)]
    test_flash(
        s1_values=s1_values,
        perf_mode=perf_mode,
    )
