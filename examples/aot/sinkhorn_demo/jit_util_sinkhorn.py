"""
Minimal PTO demo: doubly-stochastic Sinkhorn normalization (fp16, K=4).

Mirrors DeepSeek TileKernels ``sinkhorn_normalize_ref`` for verification.

PTO-ISA headers are resolved in this order:

1. ``$PTO_LIB_PATH`` (``.../include`` is appended when pointing at a repo root).
2. ``/sources/pto-isa/include`` when present (typical team image layout).
3. ``<pto-dsl>/build/_deps/libpto_isa_headers-src/include`` after a CMake FetchContent build.
"""
import ctypes
import os
import subprocess
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
# ``examples/aot/sinkhorn_demo`` → pto-dsl repository root
_PTO_DSL_ROOT = _HERE.parents[3]


def _resolve_pto_include_dir() -> str:
    env = os.environ.get("PTO_LIB_PATH")
    if env:
        root = Path(env)
        inc = root / "include" if (root / "include" / "pto" / "pto-inst.hpp").is_file() else root
        if (inc / "pto" / "pto-inst.hpp").is_file():
            return str(inc)
    docker_default = Path("/sources/pto-isa/include")
    if (docker_default / "pto" / "pto-inst.hpp").is_file():
        return str(docker_default)
    vendored = _PTO_DSL_ROOT / "build" / "_deps" / "libpto_isa_headers-src" / "include"
    if (vendored / "pto" / "pto-inst.hpp").is_file():
        return str(vendored)
    raise RuntimeError(
        "Could not find PTO-ISA headers (pto-inst.hpp). Set PTO_LIB_PATH to a pto-isa checkout, "
        "or populate headers under pto-dsl/build/_deps via CMake FetchContent."
    )


_PTO_INCLUDE_DIR = _resolve_pto_include_dir()


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


def _emit_pto_cpp() -> None:
    """Regenerate MLIR/C++ from ``sinkhorn_k4_builder.py`` (requires editable ``pto-dsl``)."""
    out_dir = _HERE / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pto_path = out_dir / "sinkhorn_k4.pto"
    cpp_path = out_dir / "sinkhorn_k4_generated.cpp"
    builder = _HERE / "sinkhorn_k4_builder.py"
    with open(pto_path, "w") as pto_file:
        subprocess.run(
            ["python3", str(builder)],
            cwd=str(_HERE),
            check=True,
            stdout=pto_file,
            text=True,
        )
    subprocess.run(
        [
            "ptoas",
            "--enable-insert-sync",
            str(pto_path),
            "-o",
            str(cpp_path),
        ],
        check=True,
    )


def _compile_kernel() -> ctypes.CDLL:
    """Build ``call_sinkhorn`` from PTODSL-generated kernel + ``caller_sinkhorn_k4.cpp``."""
    so = _HERE / "outputs" / "kernel_sinkhorn.so"
    so.parent.mkdir(parents=True, exist_ok=True)
    _emit_pto_cpp()
    subprocess.run(
        [
            "bisheng",
            "-fPIC",
            "-shared",
            "-xcce",
            "-DMEMORY_BASE",
            "-O2",
            "-std=c++17",
            "-Wno-ignored-attributes",
            "--cce-aicore-arch=dav-c220-vec",
            "-isystem",
            _PTO_INCLUDE_DIR,
            str(_HERE / "caller_sinkhorn_k4.cpp"),
            "-o",
            str(so),
        ],
        cwd=str(_HERE),
        check=True,
    )
    lib = ctypes.CDLL(str(so))
    lib.call_sinkhorn.argtypes = _KERNEL_ARGTYPES
    lib.call_sinkhorn.restype = None
    return lib


_lib = None


def _run_kernel(x: torch.Tensor, out: torch.Tensor, repeat: int, eps: float) -> None:
    global _lib
    if _lib is None:
        _lib = _compile_kernel()
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
