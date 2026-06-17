"""
The device kernels are loaded from ``outputs/kernel_sinkhorn.so`` (batched
BATCH=8 stack load), ``outputs/kernel_sinkhorn_naive.so`` (one matrix per
vector iteration), and ``outputs/kernel_sinkhorn_v2.so`` (v2-style interleaved
column normalize). Build them via ``compile.sh``.
"""

import ctypes
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_KERNEL_SO_BATCHED = _HERE / "outputs" / "kernel_sinkhorn.so"
_KERNEL_SO_NAIVE = _HERE / "outputs" / "kernel_sinkhorn_naive.so"
_KERNEL_SO_V2 = _HERE / "outputs" / "kernel_sinkhorn_v2.so"

_KERNEL_ARGTYPES = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_float,
]


def sinkhorn_normalize_ref(
    x: torch.Tensor, repeat: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    """Exact copy of ``sinkhorn_normalize_ref`` from deepseek-ai/TileKernels."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def bandwidth_gbs(data_bytes: int, duration_us: float) -> float:
    """Effective GB/s (decimal 1e9), matching ``bench_common.bandwidth_gbs``."""
    return (data_bytes / 1e9) / (duration_us / 1e6) if duration_us > 0 else 0.0


def _kernel_so_missing_message(path: Path) -> str:
    return (
        f"Kernel shared library not found: {path}\n"
        "Build kernels from this example directory, for example:\n"
        "  cd examples/aot/sinkhorn_demo && ./compile.sh"
    )


_libs: dict[str, ctypes.CDLL] = {}


def _load_kernel(so_path: Path) -> ctypes.CDLL:
    if not so_path.is_file():
        raise FileNotFoundError(_kernel_so_missing_message(so_path))
    key = str(so_path)
    if key not in _libs:
        lib = ctypes.CDLL(str(so_path))
        lib.call_sinkhorn.argtypes = _KERNEL_ARGTYPES
        lib.call_sinkhorn.restype = None
        _libs[key] = lib
    return _libs[key]


def load_sinkhorn_host_lib(so_path: Path) -> ctypes.CDLL:
    """Load any shared library exporting ``call_sinkhorn`` (same ABI as this demo)."""
    return _load_kernel(so_path)


def _run_kernel(
    lib: ctypes.CDLL,
    x: torch.Tensor,
    out: torch.Tensor,
    repeat: int,
    eps: float,
) -> None:
    dev = torch.npu.current_device()
    lib.call_sinkhorn(
        torch.npu.get_device_properties(dev).cube_core_num,
        torch.npu.current_stream()._as_parameter_,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        x.numel() // (4 * 4),
        repeat,
        float(eps),
    )


def sinkhorn_normalize(
    x: torch.Tensor,
    repeat: int = 10,
    eps: float = 1e-6,
    *,
    impl: str = "batched",
) -> torch.Tensor:
    """Run the PTO kernel (forward only). ``x`` must be fp16 on NPU, shape ``(..., 4, 4)``.

    impl:
      ``"batched"`` — ``sinkhorn_batch8_builder.py`` (stacked load, matches jit C++ demo).
      ``"naive"`` — ``sinkhorn_k4_builder.py`` (one matrix per inner iteration).
      ``"v2"`` — ``sinkhorn_v2_builder.py`` (interleaved col norm for large ``N``).
    """
    assert x.dtype == torch.float16, "demo requires fp16"
    assert x.shape[-2:] == (4, 4), "demo supports K=4 only"
    if impl == "batched":
        so = _KERNEL_SO_BATCHED
    elif impl == "naive":
        so = _KERNEL_SO_NAIVE
    elif impl == "v2":
        so = _KERNEL_SO_V2
    else:
        raise ValueError(f"impl must be 'batched', 'naive', or 'v2', got {impl!r}")

    x_flat = x.reshape(-1, 4, 4).contiguous()
    out_flat = torch.empty_like(x_flat)
    _run_kernel(_load_kernel(so), x_flat, out_flat, repeat, eps)
    torch.npu.synchronize()
    return out_flat.reshape_as(x)


def bench_sinkhorn_forward_gbs(
    x: torch.Tensor,
    repeat: int,
    eps: float,
    *,
    impl: str | None = None,
    lib: ctypes.CDLL | None = None,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Median effective GB/s for one forward pass (read input + write output).

    Pass exactly one of ``impl`` (``"batched"``, ``"naive"``, or ``"v2"`` —
    loads this demo's ``outputs/*.so``) or ``lib`` (any ``call_sinkhorn`` host
    ABI, e.g. hand-written ``cpp_ref/kernel_sinkhorn.cpp`` in this demo).
    """
    if lib is None:
        if impl is None:
            raise ValueError("bench_sinkhorn_forward_gbs requires impl or lib")
        if impl == "batched":
            so = _KERNEL_SO_BATCHED
        elif impl == "naive":
            so = _KERNEL_SO_NAIVE
        elif impl == "v2":
            so = _KERNEL_SO_V2
        else:
            raise ValueError(impl)
        lib = _load_kernel(so)
    elif impl is not None:
        raise ValueError("pass only one of impl and lib")

    x_flat = x.reshape(-1, 4, 4).contiguous()
    out_flat = torch.empty_like(x_flat)
    nmat = x_flat.shape[0]
    bytes_rw = 2 * nmat * 4 * 4 * 2  # read fp16 + write fp16

    for _ in range(warmup):
        _run_kernel(lib, x_flat, out_flat, repeat, eps)
    torch.npu.synchronize()

    times_us: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _run_kernel(lib, x_flat, out_flat, repeat, eps)
        torch.npu.synchronize()
        times_us.append((time.perf_counter() - t0) * 1e6)

    times_us.sort()
    med_us = times_us[len(times_us) // 2]
    return bandwidth_gbs(bytes_rw, med_us)
