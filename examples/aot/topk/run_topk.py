"""
Run and validate the TopK AOT kernel for multiple configurations.

The kernel is **dynamic** in n_rows: a single compiled .so handles any row
count.  Configs that share the same (n_cols, topk) pair reuse the same library.

Usage:
    python ./run_topk.py              # compile + run all configs
    python ./run_topk.py --no-compile # skip recompilation (libs already built)

Valid N_COLS values (SORT_BLOCK_LEN=32)
---------------------------------------
  SORT_COLS = N_COLS*2 must be a power-of-4 multiple of HW_BLOCK_LEN=64:
    N_COLS =  128 → 1 merge pass
    N_COLS =  512 → 2 merge passes
    N_COLS = 2048 → 3 merge passes
"""

import argparse
import ctypes
import os
import subprocess

import torch
import torch_npu

from ptodsl.test_util import get_test_device
from topk_builder import fn_name

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── test configurations ───────────────────────────────────────────────────────
# (n_rows, n_cols, topk, description)
# n_rows can be any positive integer – divisibility by BLOCK_DIM is NOT required.
# Configs sharing the same (n_cols, topk) reuse the same compiled .so.
_CONFIGS = [
    # 1 merge pass – topk < n_cols
    (24, 128, 64, "n_rows=24,  1 pass, topk<n_cols"),
    (48, 128, 64, "n_rows=48,  1 pass, topk<n_cols – same .so"),
    (37, 128, 64, "n_rows=37,  1 pass, non-multiple of 24"),
    # 1 merge pass – topk = n_cols
    (48, 128, 128, "n_rows=48,  1 pass, topk=n_cols"),
    # 2 merge passes – topk < n_cols
    (48, 512, 128, "n_rows=48,  2 passes, topk<n_cols"),
    (4800, 512, 256, "n_rows=4800, 2 passes, reference shape"),
    (240, 512, 256, "n_rows=240, 2 passes, same .so as reference"),
    (25, 512, 256, "n_rows=25,  2 passes, non-multiple of 24"),
    # 2 merge passes – topk = n_cols
    (48, 512, 512, "n_rows=48,  2 passes, topk=n_cols"),
    # 3 merge passes – topk < n_cols
    (48, 2048, 512, "n_rows=48,  3 passes, topk<n_cols"),
    # 3 merge passes – topk = n_cols
    (48, 2048, 2048, "n_rows=48,  3 passes, topk=n_cols"),
]


def _lib_path(n_cols: int, topk: int) -> str:
    return os.path.join(_DIR, f"{fn_name(n_cols, topk)}_lib.so")


def _compile(n_cols: int, topk: int) -> None:
    subprocess.check_call(
        ["bash", os.path.join(_DIR, "compile.sh"), str(n_cols), str(topk)],
        cwd=_DIR,
    )


def _load_fn(n_cols: int, topk: int):
    lib = ctypes.CDLL(_lib_path(n_cols, topk))
    fn = getattr(lib, f"call_{fn_name(n_cols, topk)}")
    fn.argtypes = [
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # src         [n_rows, n_cols] float32
        ctypes.c_void_p,  # inIdx       [n_cols]         uint32
        ctypes.c_void_p,  # out_scores  [n_rows, topk]   float32
        ctypes.c_void_p,  # out_indices [n_rows, topk]   uint32
        ctypes.c_int32,  # n_rows (runtime)
    ]
    fn.restype = None
    return fn


def _ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _run_one(device: str, n_rows: int, n_cols: int, topk: int, desc: str) -> None:
    fn = _load_fn(n_cols, topk)
    torch.manual_seed(0)

    src = torch.rand(n_rows, n_cols, dtype=torch.float32, device=device)
    inidx = torch.arange(n_cols, dtype=torch.int32, device=device)
    out_scores = torch.empty(n_rows, topk, dtype=torch.float32, device=device)
    out_indices = torch.empty(n_rows, topk, dtype=torch.int32, device=device)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    torch.npu.synchronize()
    fn(
        stream_ptr,
        _ptr(src),
        _ptr(inidx),
        _ptr(out_scores),
        _ptr(out_indices),
        ctypes.c_int32(n_rows),
    )
    torch.npu.synchronize()

    src_cpu = src.cpu()

    # 1. Scores must exactly match torch.topk (descending, sorted).
    ref_vals, _ = torch.topk(src_cpu, topk, dim=-1, largest=True, sorted=True)
    torch.testing.assert_close(
        out_scores.cpu(),
        ref_vals,
        rtol=0,
        atol=0,
        msg=f"scores mismatch ({desc})",
    )

    # 2. Each returned index must point to the correct value in the source row.
    #    (Don't compare indices directly – hardware may break ties differently.)
    gathered = torch.gather(src_cpu, 1, out_indices.cpu().to(torch.int64))
    torch.testing.assert_close(
        gathered,
        out_scores.cpu(),
        rtol=0,
        atol=0,
        msg=f"index↔score mismatch ({desc})",
    )

    print(f"  PASSED  {n_rows:5d}×{n_cols:5d} → top-{topk:5d}  [{desc}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip recompilation (assume .so files already exist)",
    )
    args = parser.parse_args()

    device = get_test_device()
    torch.npu.set_device(device)

    print(f"Running {len(_CONFIGS)} TopK configs on {device}")
    print("-" * 70)

    compiled: set = set()
    for n_rows, n_cols, topk, desc in _CONFIGS:
        if not args.no_compile and (n_cols, topk) not in compiled:
            _compile(n_cols, topk)
            compiled.add((n_cols, topk))
        _run_one(device, n_rows, n_cols, topk, desc)

    print("-" * 70)
    print(f"All {len(_CONFIGS)} configs PASSED.")


if __name__ == "__main__":
    main()
