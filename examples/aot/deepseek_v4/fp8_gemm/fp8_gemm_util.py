"""Reference + ctypes wrapper for the deepseek_v4 ``fp8_gemm`` PTO kernel.

Scale-fusion semantics — host-side pre-scale design
---------------------------------------------------

The GPU TileLang kernel performs::

    C[m, n] = sum_k A[m, k] * B[k, n] * Sa[m, k_g] * Sb[k_g, n_b]

where ``k_g = k // BLOCK_K`` and ``n_b = n // BLOCK_N`` index into per-block
scale tensors. On GPU this fusion is a hard requirement because FP8 has a
tiny dynamic range (±240) and pre-multiplying ``A`` by ``Sa`` would saturate
the input.

On the NPU we use **FP16** activations / weights, whose ±65504 range
comfortably accommodates pre-scaled values for any realistic ``Sa, Sb``.
We therefore pre-scale ``A`` and ``B`` host-side, then run a plain FP16
GEMM on-device. Mathematically::

    A_scaled[m, k] = A[m, k] * Sa[m, k_g]                  (row-broadcast)
    B_scaled[k, n] = B[k, n] * Sb[k_g, n_b]                (col-broadcast)
    C[m, n]        = sum_k A_scaled[m, k] * B_scaled[k, n]

is identical to the GPU formulation. The kernel itself is unchanged —
``Sa, Sb`` are still GEMM inputs to keep the API contract, but they are
applied during ``fp8_gemm()`` before the kernel call. This is the
NPU-equivalent of "scale fusion".
"""

import ctypes
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
_KERNEL_SO = _HERE / "fp8_gemm_lib.so"

BLOCK_M = 32
BLOCK_N = 128
BLOCK_K = 128


def _prescale(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor):
    """Apply the per-block scales to A and B in fp32, then cast back to fp16.

    Sa[M, K/BLOCK_K]      -> row-broadcast over each K-group.
    Sb[K/BLOCK_K, N/BLOCK_N] -> col-broadcast over each (K-group, N-block).
    """
    M, K = a.shape
    _, N = b.shape
    Kg, Nb = sb.shape
    assert sa.shape == (M, Kg)

    # Sa: [M, Kg] -> [M, Kg, 1] -> [M, K] via expand on the BLOCK_K axis.
    sa_exp = sa.unsqueeze(-1).expand(M, Kg, BLOCK_K).reshape(M, K)
    # Sb: [Kg, Nb] -> [Kg, 1, Nb, 1] -> [K, N] expanding on BLOCK_K, BLOCK_N.
    sb_exp = (
        sb.unsqueeze(1).unsqueeze(-1).expand(Kg, BLOCK_K, Nb, BLOCK_N).reshape(K, N)
    )

    a_s = (a.to(torch.float32) * sa_exp).to(torch.float16)
    b_s = (b.to(torch.float32) * sb_exp).to(torch.float16)
    return a_s, b_s


def fp8_gemm_ref(
    a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    """Full-fidelity reference: fp32 multiply-accumulate then cast to fp16."""
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


def fp8_gemm(
    a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor
) -> torch.Tensor:
    """Run kernel with full per-block scale fusion (host pre-scale).

    Args:
        a: [M, K]                 fp16, NPU
        b: [K, N]                 fp16, NPU
        sa: [M, K // BLOCK_K]     fp32, NPU
        sb: [K // BLOCK_K, N // BLOCK_N] fp32, NPU
    """
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
