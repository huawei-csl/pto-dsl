"""Reference + ctypes wrapper for the deepseek_v4 ``fp4_gemm`` PTO kernel.

Same host-side pre-scale design as ``fp8_gemm`` (see
``fp8_gemm_util.py`` for full rationale). Only differences:

* ``BLOCK_K = 32`` (matches the GPU FP4 weight group size).
* Sa shape  ``[M, K // 32]``;  Sb shape ``[K // 32, N // 128]``.

NPU FP16 dynamic range comfortably accommodates pre-scaled values, so we
absorb both scales into ``A`` and ``B`` host-side before launching the
kernel — mathematically identical to GPU per-block fusion.
"""

import ctypes
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
_KERNEL_SO = _HERE / "fp4_gemm_lib.so"

BLOCK_M = 32
BLOCK_N = 128
BLOCK_K = 32  # weight group


def _prescale(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor):
    M, K = a.shape
    _, N = b.shape
    Kg, Nb = sb.shape
    assert sa.shape == (M, Kg)
    sa_exp = sa.unsqueeze(-1).expand(M, Kg, BLOCK_K).reshape(M, K)
    sb_exp = (
        sb.unsqueeze(1).unsqueeze(-1).expand(Kg, BLOCK_K, Nb, BLOCK_N).reshape(K, N)
    )
    a_s = (a.to(torch.float32) * sa_exp).to(torch.float16)
    b_s = (b.to(torch.float32) * sb_exp).to(torch.float16)
    return a_s, b_s


def fp4_gemm_ref(
    a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    a_s, b_s = _prescale(a, b, sa, sb)
    return (a_s.to(torch.float32) @ b_s.to(torch.float32)).to(torch.float16)


_ARGTYPES = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
]


_lib = None


def _load():
    global _lib
    if _lib is None:
        if not _KERNEL_SO.is_file():
            raise FileNotFoundError(
                f"Kernel shared library not found: {_KERNEL_SO}\n"
                f"Build first: cd {_HERE} && ./compile.sh"
            )
        _lib = ctypes.CDLL(str(_KERNEL_SO))
        _lib.call_kernel.argtypes = _ARGTYPES
        _lib.call_kernel.restype = None
    return _lib


def fp4_gemm(
    a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    assert a.is_npu and b.is_npu and a.dtype == b.dtype == torch.float16
    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0

    a_s, b_s = _prescale(a, b, sa, sb)
    a_s = a_s.contiguous()
    b_s = b_s.contiguous()

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    lib = _load()
    dev = torch.npu.current_device()
    blk = torch.npu.get_device_properties(dev).cube_core_num
    lib.call_kernel(
        blk,
        torch.npu.current_stream()._as_parameter_,
        ctypes.c_void_p(a_s.data_ptr()),
        ctypes.c_void_p(b_s.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.c_void_p(sa.contiguous().data_ptr()),  # not read by kernel
        ctypes.c_void_p(sb.contiguous().data_ptr()),  # not read by kernel
        ctypes.c_int32(M),
        ctypes.c_int32(N),
        ctypes.c_int32(K),
    )
    torch.npu.synchronize()
    return c
