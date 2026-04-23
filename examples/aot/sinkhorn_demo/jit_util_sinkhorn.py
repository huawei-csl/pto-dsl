"""
The device kernel is loaded from ``outputs/kernel_sinkhorn.so``, which must be
built first (see ``compile.sh`` in this directory).
"""
import ctypes
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
_KERNEL_SO = _HERE / "outputs" / "kernel_sinkhorn.so"


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Exact copy of ``sinkhorn_normalize_ref`` from deepseek-ai/TileKernels."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


_KERNEL_ARGTYPES = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_float,
]


def _kernel_so_missing_message() -> str:
    return (
        f"Kernel shared library not found: {_KERNEL_SO}\n"
        "Build it first from this example directory, for example:\n"
        "  cd examples/aot/sinkhorn_demo && ./compile.sh"
    )


def _load_kernel() -> ctypes.CDLL:
    """Load ``call_sinkhorn`` from a pre-built ``outputs/kernel_sinkhorn.so``."""
    if not _KERNEL_SO.is_file():
        raise FileNotFoundError(_kernel_so_missing_message())
    lib = ctypes.CDLL(str(_KERNEL_SO))
    lib.call_sinkhorn.argtypes = _KERNEL_ARGTYPES
    lib.call_sinkhorn.restype = None
    return lib


_lib = None


def _run_kernel(x: torch.Tensor, out: torch.Tensor, repeat: int, eps: float) -> None:
    global _lib
    if _lib is None:
        _lib = _load_kernel()
    dev = torch.npu.current_device()
    _lib.call_sinkhorn(
        torch.npu.get_device_properties(dev).cube_core_num,
        torch.npu.current_stream()._as_parameter_,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        x.numel() // (4 * 4),
        repeat,
        float(eps),
    )


def sinkhorn_normalize(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Run the PTO kernel (forward only). ``x`` must be fp16 on NPU, shape ``(..., 4, 4)``."""
    assert x.dtype == torch.float16, "demo requires fp16"
    assert x.shape[-2:] == (4, 4), "demo supports K=4 only"
    x_flat = x.reshape(-1, 4, 4).contiguous()
    out_flat = torch.empty_like(x_flat)
    _run_kernel(x_flat, out_flat, repeat, eps)
    torch.npu.synchronize()
    return out_flat.reshape_as(x)
