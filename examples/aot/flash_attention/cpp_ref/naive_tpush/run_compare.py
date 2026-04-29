#!/usr/bin/python3

import warnings

warnings.filterwarnings("ignore")
import ctypes
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from ptodsl.bench import do_bench
from ptodsl.utils import get_test_device

os.environ.setdefault("PTO_LIB_PATH", "/sources/pto-isa")
from jit_util_flash import jit_compile_flash  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
S0, S1, HEAD = 128 * 24, 16384, 128
WARMUP, ITERS = 10, 15
FLOPS = 4 * S0 * S1 * HEAD + S0 * (6 * S1 - 2)


def ptr(t):
    return ctypes.c_void_p(t.data_ptr())


def tflops(ms):
    return FLOPS / (ms * 1e-3) / 1e12


def torch_ref(q, k, v):
    return torch.softmax(q.float() @ k.float().T / math.sqrt(HEAD), dim=-1) @ v.float()


def check(name, got, ref):
    mismatches = (~torch.isclose(got, ref, rtol=1e-3, atol=1e-3)).sum().item()
    if mismatches:
        raise SystemExit(f"{name}: {mismatches} elements mismatch")


def load_ptodsl():
    subprocess.run(
        ["bash", str(THIS_DIR / "compile.sh")],
        cwd=THIS_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    sys.path.insert(0, str(THIS_DIR / "kernels"))
    import fa_dsl_builder as b  # noqa: E402

    lib = ctypes.CDLL(str(THIS_DIR / "build_artifacts" / "fa_dsl.so"))
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

    block_dim = S0 // b.CUBE_S0
    gm_slot = torch.empty(
        (b.GM_ELEMS_PER_BLOCK * block_dim,), dtype=torch.float32, device="npu"
    )
    out = torch.empty((S0, HEAD), dtype=torch.float32, device="npu")

    def flash(q, k, v):
        lib.call_kernel(
            block_dim,
            torch.npu.current_stream()._as_parameter_,
            ptr(gm_slot),
            ptr(q),
            ptr(k),
            ptr(v),
            ptr(out),
            S0,
            S1,
        )
        return out

    return flash


def main():
    torch.npu.set_device(get_test_device())
    torch.manual_seed(1)
    torch.npu.manual_seed(1)

    q = torch.randn((S0, HEAD), dtype=torch.float16, device="npu")
    k = torch.randn((S1, HEAD), dtype=torch.float16, device="npu")
    v = torch.randn((S1, HEAD), dtype=torch.float16, device="npu")

    ptodsl = load_ptodsl()
    cpp = jit_compile_flash(kernel_cpp=str(THIS_DIR / "fa_kernel.cpp"), verbose=False)
    ref = torch_ref(q, k, v)
    check("PTODSL", ptodsl(q, k, v), ref)
    check("fa_kernel.cpp", cpp(q, k, v), ref)

    for name, fn in (("PTODSL", ptodsl), ("fa_kernel.cpp", cpp)):
        ms = do_bench(
            lambda: fn(q, k, v), warmup_iters=WARMUP, benchmark_iters=ITERS, unit="ms"
        )
        print(f"{name}: {tflops(ms):.3f} TFLOP/s")


if __name__ == "__main__":
    main()
