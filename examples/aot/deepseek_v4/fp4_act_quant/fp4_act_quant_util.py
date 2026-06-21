"""Reference + ctypes wrapper for the deepseek_v4 ``fp4_act_quant`` PTO kernel."""

import ctypes
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
_KERNEL_SO = _HERE / "fp4_act_quant_lib.so"

BLOCK_SIZE = 32
FP4_MAX = 6.0  # max representable magnitude of FP4 e2m1


def fp4_act_quant_ref(x: torch.Tensor, block_size: int = BLOCK_SIZE):
    """Per-row-block symmetric FP4-style int quant.

    Returns ``(y_int8 [M, N] in [-7,7], s_fp32 [M, N // block_size])``.
    The NPU port stores the FP4 codes packed in int8 (one per byte).
    """
    assert x.dtype == torch.float16
    assert x.dim() == 2 and x.shape[1] % block_size == 0
    M, N = x.shape
    nb = N // block_size

    x_f32 = x.to(torch.float32).reshape(M, nb, block_size)
    amax = x_f32.abs().amax(dim=-1)
    scale = (amax / FP4_MAX).clamp(min=1e-12)
    y = (x_f32 / scale.unsqueeze(-1)).round().clamp(-7, 7)
    return y.to(torch.int8).reshape(M, N), scale.to(torch.float32)


_ARGTYPES = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
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


def fp4_act_quant(x: torch.Tensor):
    assert x.is_npu and x.dtype == torch.float16
    M, N = x.shape
    assert N % BLOCK_SIZE == 0
    y = torch.empty((M, N), dtype=torch.int8, device=x.device)
    s_storage = torch.empty((N // BLOCK_SIZE, M), dtype=torch.float32, device=x.device)
    s = s_storage.t()
    lib = _load()
    dev = torch.npu.current_device()
    blk = torch.npu.get_device_properties(dev).cube_core_num
    lib.call_kernel(
        blk,
        torch.npu.current_stream()._as_parameter_,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_void_p(s.data_ptr()),
        ctypes.c_int32(M),
        ctypes.c_int32(N),
    )
    torch.npu.synchronize()
    return y, s
