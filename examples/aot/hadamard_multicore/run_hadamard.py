"""Fast Walsh-Hadamard Transform — dynamic gather-index variant.

Compile and run:
    python run_hadamard.py
"""

import ctypes
import os
import subprocess

import torch
import torch_npu

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = "npu:6"

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
}


def lib_path(dtype):
    return os.path.join(_DIR, f"{dtype}_dynamic_lib.so")


def compile_kernel(dtype):
    subprocess.check_call(
        ["bash", os.path.join(_DIR, "compile.sh"), dtype],
        cwd=_DIR,
    )


def fwht_ref(x: torch.Tensor) -> torch.Tensor:
    """log2(n)-stage loop: P0101 gather for even elements, P1010 gather for odd elements."""
    x = x.clone()
    n = x.shape[-1]
    log2_n = n.bit_length() - 1
    flat = x.view(-1, n)
    for _ in range(log2_n):
        even = flat[:, 0::2].clone()    # P0101 gather
        odd  = flat[:, 1::2].clone()    # P1010 gather
        flat[:, :n // 2] = even + odd
        flat[:, n // 2:] = even - odd
    return x


def run(dtype: str, batch: int, n: int) -> None:
    torch.npu.set_device(_DEVICE)
    torch_dtype = TORCH_DTYPES[dtype]

    compile_kernel(dtype)
    lib = ctypes.CDLL(lib_path(dtype))
    fn = getattr(lib, f"call_{dtype}_dynamic")

    stream_ptr = torch.npu.current_stream()._as_parameter_

    x     = torch.rand(batch, n, device=_DEVICE, dtype=torch_dtype)
    x_ref = x.clone()
    out   = torch.empty_like(x, device=_DEVICE, dtype=torch_dtype)

    torch.npu.synchronize()
    fn(
        stream_ptr,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int32(batch),
        ctypes.c_int32(n),
        ctypes.c_int32(n.bit_length() - 1),
    )
    torch.npu.synchronize()

    ref = fwht_ref(x_ref)
    print(out)
    print(ref)
    torch.testing.assert_close(out, ref.to(torch_dtype), rtol=1e-2, atol=1e-2)
    print(f"  PASS  dtype={dtype}  n={n}  batch={batch}")
    # library is reused across shapes — don't remove it here


if __name__ == "__main__":
    cases = [
        ("float16", 1, 32),
        ("float16", 4,  32),
        ("float16", 8,  32),
        ("float16", 13,  32),
    ]
    print("Fast Walsh-Hadamard Transform — multi-core NPU (dynamic gather)")
    for dtype, batch, n in cases:
        run(dtype, batch, n)
    print("All cases passed.")
