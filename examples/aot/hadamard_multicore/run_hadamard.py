"""Fast Walsh-Hadamard Transform example.

Compile for a given (dtype, n) first:
    bash compile.sh float16 32

Then run:
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


def case_id(dtype, n):
    return f"{dtype}_n{n}"


def lib_path(dtype, n):
    return os.path.join(_DIR, f"{case_id(dtype, n)}_lib.so")


def compile_kernel(dtype, n):
    subprocess.check_call(
        ["bash", os.path.join(_DIR, "compile.sh"), dtype, str(n)],
        cwd=_DIR,
    )


def fwht_ref(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard Transform reference.

    Mirrors the kernel's butterfly exactly:
      each stage gathers even positions (0,2,4,...) and odd positions (1,3,5,...),
      then writes [sums | diffs]: sums to the first half, diffs to the second half.
    """
    x = x.clone()
    n = x.shape[-1]
    log2_n = n.bit_length() - 1
    flat = x.view(-1, n)
    for _ in range(log2_n):
        even = flat[:, 0::2].clone()   # positions 0, 2, 4, ... (P0101)
        odd  = flat[:, 1::2].clone()   # positions 1, 3, 5, ... (index gather)
        flat[:, :n // 2] = even + odd  # sums → first half
        flat[:, n // 2:] = even - odd  # diffs → second half
    return x


def run(dtype: str, n: int, batch: int = 32) -> None:
    torch.npu.set_device(_DEVICE)
    torch_dtype = TORCH_DTYPES[dtype]

    compile_kernel(dtype, n)
    lib = ctypes.CDLL(lib_path(dtype, n))
    fn = getattr(lib, f"call_{case_id(dtype, n)}")

    stream_ptr = torch.npu.current_stream()._as_parameter_

    x     = torch.rand(batch, n, device=_DEVICE, dtype=torch_dtype)  # input data
    out = torch.empty_like(x,  device=_DEVICE,dtype=torch_dtype)  # output buffer
    x_ref = x.clone()

    torch.npu.synchronize()
    fn(
        stream_ptr,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
    )
    torch.npu.synchronize()

    ref = fwht_ref(x_ref)
    print(out)
    print(ref)
    torch.testing.assert_close(
        out, ref, rtol=1e-2, atol=1e-2
    )
    print(f"  PASS  dtype={dtype}  n={n}  batch={batch}")
    os.remove(lib_path(dtype, n))


if __name__ == "__main__":
    cases = [
        ("float32", 32),
        #("float32", 64),
    ]
    print("Fast Walsh-Hadamard Transform — multi-core NPU")
    for dtype, n in cases:
        run(dtype, n)
    print("All cases passed.")
