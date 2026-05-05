"""Reference + ctypes wrapper for the deepseek_v4 ``act_quant`` PTO kernel.

Reference matches the GPU TileLang behaviour adapted to the NPU port:
FP16 input, int8 output, FP32 per-row-block reciprocal scale, K-group=128.
"""

import ctypes
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
_KERNEL_SO = _HERE / "act_quant_lib.so"

BLOCK_SIZE = 128
INT8_MAX = 127.0


def act_quant_ref(x: torch.Tensor, block_size: int = BLOCK_SIZE):
    """Reference: per-row-block symmetric int8 quant.

    ``x``: [M, N] fp16, N % block_size == 0.
    Returns ``(y_int8 [M, N], s_fp32 [M, N // block_size])`` on the same device.
    """
    assert x.dtype == torch.float16, "fp16 input expected"
    assert x.dim() == 2 and x.shape[1] % block_size == 0
    M, N = x.shape
    nb = N // block_size

    x_f32 = x.to(torch.float32).reshape(M, nb, block_size)
    amax = x_f32.abs().amax(dim=-1, keepdim=False)  # [M, nb]
    scale = (amax / INT8_MAX).clamp(min=1e-12)  # avoid /0
    y = (x_f32 / scale.unsqueeze(-1)).round().clamp(-127, 127)
    y_i8 = y.to(torch.int8).reshape(M, N)
    return y_i8, scale.to(torch.float32)


_ARGTYPES = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
]


def _missing_msg() -> str:
    return (
        f"Kernel shared library not found: {_KERNEL_SO}\n"
        "Build it first:\n"
        f"  cd {_HERE} && ./compile.sh"
    )


_lib = None


def _load():
    global _lib
    if _lib is None:
        if not _KERNEL_SO.is_file():
            raise FileNotFoundError(_missing_msg())
        _lib = ctypes.CDLL(str(_KERNEL_SO))
        _lib.call_kernel.argtypes = _ARGTYPES
        _lib.call_kernel.restype = None
    return _lib


def act_quant(x: torch.Tensor):
    """Run the PTO kernel. ``x``: [M, N] fp16 NPU tensor; N % BLOCK_SIZE == 0."""
    assert x.is_npu and x.dtype == torch.float16
    M, N = x.shape
    assert N % BLOCK_SIZE == 0
    y = torch.empty((M, N), dtype=torch.int8, device=x.device)
    # Kernel writes scale in COL-MAJOR layout (strides=[1, M]).
    # Allocate as a transpose of a contiguous [N//B, M] tensor.
    s_storage = torch.empty((N // BLOCK_SIZE, M), dtype=torch.float32, device=x.device)
    s = s_storage.t()  # logical shape [M, N//BLOCK_SIZE], strides [1, M]
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
