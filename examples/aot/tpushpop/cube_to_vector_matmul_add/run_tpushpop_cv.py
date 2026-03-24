import ctypes
import os
import subprocess

import numpy as np
import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB_PATH = os.path.join(THIS_DIR, "tpushpop_cv_lib.so")
DEFAULT_COMPILE_SCRIPT = os.path.join(THIS_DIR, "compile.sh")
DEFAULT_FIFO_BYTES = 4 * 1024
TOTAL_M = 16
K = 32
N = 32
INPUT_DTYPE = torch.float16
SEED = 0
ATOL = 5e-2
RTOL = 5e-2
SANITY_ONLY = False


def torch_to_ctypes(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def compile_example(compile_script: str) -> None:
    subprocess.run(
        ["bash", compile_script],
        check=True,
        cwd=THIS_DIR,
    )


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


def make_buffers(
    *,
    total_m: int,
    k: int,
    n: int,
    input_dtype: torch.dtype,
    device: str,
    fifo_bytes: int,
):
    src_a = torch.randn((total_m, k), dtype=input_dtype, device=device)
    src_b = torch.randn((k, n), dtype=input_dtype, device=device)
    bias = torch.randn((total_m, n), dtype=torch.float32, device=device)
    out = torch.zeros((total_m, n), dtype=torch.float32, device=device)

    fifo_elems = max(1, (fifo_bytes + 3) // 4)
    fifo_mem = torch.zeros((fifo_elems,), dtype=torch.float32, device=device)
    return out, src_a, src_b, bias, fifo_mem


def run_kernel(
    lib: ctypes.CDLL,
    *,
    out: torch.Tensor,
    src_a: torch.Tensor,
    src_b: torch.Tensor,
    bias: torch.Tensor,
    fifo_mem: torch.Tensor,
) -> torch.Tensor:
    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        1,
        stream_ptr,
        torch_to_ctypes(out),
        torch_to_ctypes(src_a),
        torch_to_ctypes(src_b),
        torch_to_ctypes(bias),
        torch_to_ctypes(fifo_mem),
    )
    torch.npu.synchronize()
    return out


def reference_result(src_a: torch.Tensor, src_b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    ref = torch.matmul(src_a.float().cpu(), src_b.float().cpu())
    if not SANITY_ONLY:
        ref = ref + bias.cpu()
    return ref.to(torch.float32)


def main() -> None:
    compile_example(DEFAULT_COMPILE_SCRIPT)

    device = get_test_device()
    torch.npu.set_device(device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    lib = load_lib(DEFAULT_LIB_PATH)
    out, src_a, src_b, bias, fifo_mem = make_buffers(
        total_m=TOTAL_M,
        k=K,
        n=N,
        input_dtype=INPUT_DTYPE,
        device=device,
        fifo_bytes=DEFAULT_FIFO_BYTES,
    )

    out = run_kernel(
        lib,
        out=out,
        src_a=src_a,
        src_b=src_b,
        bias=bias,
        fifo_mem=fifo_mem,
    )
    ref = reference_result(src_a, src_b, bias)
    out_cpu = out.cpu()

    max_abs = float(torch.max(torch.abs(out_cpu - ref)).item())
    mean_abs = float(torch.mean(torch.abs(out_cpu - ref)).item())
    ok = bool(torch.allclose(out_cpu, ref, atol=ATOL, rtol=RTOL))

    print(
        f"mode={'sanity_matmul' if SANITY_ONLY else 'tpushpop_cv'} "
        f"shape=({TOTAL_M}, {K}, {N}) dtype={INPUT_DTYPE} "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
    )

    if not ok:
        raise SystemExit(
            f"Validation failed with atol={ATOL} rtol={RTOL}. "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
        )

    print(f"Validation passed using {DEFAULT_LIB_PATH}.")


if __name__ == "__main__":
    main()
