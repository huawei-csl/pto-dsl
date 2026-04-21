"""JIT utilities for the Sinkhorn example.

Provides three entry points:

* :func:`compile_pto_lib` — Runs ``sinkhorn_builder.py`` → ``ptoas`` → ``bisheng``
  to produce ``sinkhorn_lib.so`` (the PTODSL kernel).
* :func:`compile_reference_lib` — Compiles ``reference.cpp`` directly with
  ``bisheng`` to produce ``reference_lib.so`` (the hand-tuned baseline).
* :func:`load_lib` — ``ctypes`` wrapper around ``call_sinkhorn_kernel``,
  shared by both libraries (identical C ABI).

Caches build artefacts under ``outputs/so/`` next to this file; rebuilds only
when the source file is newer than the cached ``.so``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
import torch_npu  # noqa: F401  (registers torch.npu)

from ptodsl.npu_info import get_num_cube_cores

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SO_DIR = THIS_DIR / "outputs" / "so"
PTO_LIB_PATH = Path(os.environ.get("PTO_LIB_PATH", "/sources/pto-isa"))

MAX_DIM = 256
ROW_CHUNK = 8
BLOCK_DIM = get_num_cube_cores()

SINKHORN_ARGTYPES = [
    ctypes.c_uint32,  # blockDim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # matrix_in
    ctypes.c_void_p,  # matrix_out
    ctypes.c_void_p,  # mu1_out
    ctypes.c_void_p,  # mu2_out
    ctypes.c_uint32,  # N
    ctypes.c_uint32,  # K
    ctypes.c_uint32,  # L
    ctypes.c_uint32,  # order
    ctypes.c_float,  # lr
    ctypes.c_float,  # eps
    ctypes.c_float,  # invK
    ctypes.c_float,  # invL
    ctypes.c_float,  # invK1
    ctypes.c_float,  # invL1
]

BISHENG_FLAGS = [
    "-fPIC",
    "-shared",
    "-D_FORTIFY_SOURCE=2",
    "-O2",
    "-std=c++17",
    "-Wno-macro-redefined",
    "-Wno-ignored-attributes",
    "-fstack-protector-strong",
    "-xcce",
    "-Xhost-start",
    "-Xhost-end",
    "-mllvm",
    "-cce-aicore-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-function-stack-size=0x8000",
    "-mllvm",
    "-cce-aicore-record-overflow=true",
    "-mllvm",
    "-cce-aicore-addr-transform",
    "-mllvm",
    "-cce-aicore-dcci-insert-for-scalar=false",
    "--npu-arch=dav-2201",
    "-DMEMORY_BASE",
    "-std=gnu++17",
]


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _file_digest(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _run(cmd, *, cwd=None, verbose=False):
    if verbose:
        print(f"$ {' '.join(map(str, cmd))}")
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)


def _bisheng_compile(srcs, out_so: Path, *, defines=None, verbose=False):
    out_so.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bisheng", f"-I{PTO_LIB_PATH}/include", *BISHENG_FLAGS]
    for k, v in (defines or {}).items():
        cmd.append(f"-D{k}={v}")
    cmd += [*map(str, srcs), "-o", str(out_so)]
    _run(cmd, verbose=verbose)


def compile_pto_lib(
    builder_py: Path | str = THIS_DIR / "sinkhorn_builder.py",
    *,
    so_dir: Path | str = DEFAULT_SO_DIR,
    verbose: bool = False,
    force: bool = False,
) -> Path:
    """Build the PTODSL Sinkhorn kernel into ``<so_dir>/sinkhorn_lib.so``."""
    builder_py = Path(builder_py).resolve()
    caller_cpp = THIS_DIR / "caller.cpp"
    so_dir = Path(so_dir)
    so_dir.mkdir(parents=True, exist_ok=True)

    digest = _file_digest(builder_py, caller_cpp)
    out_so = so_dir / f"sinkhorn_lib.{digest}.so"
    if out_so.exists() and not force:
        return out_so

    pto_path = so_dir / f"sinkhorn.{digest}.pto"
    cpp_path = so_dir / f"sinkhorn.{digest}.cpp"

    # 1) builder -> .pto
    if verbose:
        print(f"$ python {builder_py} > {pto_path}")
    with open(pto_path, "w") as f:
        subprocess.run([sys.executable, str(builder_py)], check=True, stdout=f)

    # 2) ptoas -> .cpp
    _run(
        ["ptoas", "--enable-insert-sync", str(pto_path), "-o", str(cpp_path)],
        verbose=verbose,
    )

    # 3) bisheng caller.cpp (which #includes the generated cpp)
    _bisheng_compile(
        [caller_cpp],
        out_so,
        defines={"KERNEL_CPP": f'\\"{cpp_path}\\"'},
        verbose=verbose,
    )
    return out_so


def compile_reference_lib(
    reference_cpp: Path | str = THIS_DIR / "reference.cpp",
    *,
    so_dir: Path | str = DEFAULT_SO_DIR,
    verbose: bool = False,
    force: bool = False,
) -> Path:
    """Build ``reference.cpp`` into ``<so_dir>/reference_lib.so``."""
    reference_cpp = Path(reference_cpp).resolve()
    so_dir = Path(so_dir)
    so_dir.mkdir(parents=True, exist_ok=True)

    digest = _file_digest(reference_cpp)
    out_so = so_dir / f"reference_lib.{digest}.so"
    if out_so.exists() and not force:
        return out_so

    _bisheng_compile([reference_cpp], out_so, verbose=verbose)
    return out_so


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _validate_io(matrix_in, matrix_out, mu1_out, mu2_out, K, L):
    if matrix_in.dim() != 3:
        raise ValueError("matrix_in must be a 3D tensor (N, K, L).")
    N = matrix_in.shape[0]
    if matrix_in.shape[1] != K or matrix_in.shape[2] != L:
        raise ValueError(f"matrix_in must have shape (N, {K}, {L}).")
    if matrix_out.shape != matrix_in.shape:
        raise ValueError("matrix_out must have the same shape as matrix_in.")
    if mu1_out.shape != (N, L):
        raise ValueError(f"mu1_out must have shape ({N}, {L}).")
    if mu2_out.shape != (N, K):
        raise ValueError(f"mu2_out must have shape ({N}, {K}).")
    for name, t in [
        ("matrix_in", matrix_in),
        ("matrix_out", matrix_out),
        ("mu1_out", mu1_out),
        ("mu2_out", mu2_out),
    ]:
        if t.dtype != torch.float16:
            raise TypeError(f"{name} must use torch.float16.")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    if not (matrix_in.device == matrix_out.device == mu1_out.device == mu2_out.device):
        raise ValueError("All tensors must be on the same device.")
    if K > MAX_DIM or L > MAX_DIM:
        raise ValueError(f"K and L must be <= {MAX_DIM}.")
    if K == 0 or L == 0:
        raise ValueError("K and L must be positive.")


def load_lib(lib_path: Path | str, *, block_dim: int = BLOCK_DIM) -> Callable:
    """Open ``lib_path`` and return a ``sinkhorn(...)`` callable."""
    lib = ctypes.CDLL(str(lib_path))
    lib.call_sinkhorn_kernel.argtypes = SINKHORN_ARGTYPES
    lib.call_sinkhorn_kernel.restype = None
    block_dim = max(1, int(block_dim))

    def sinkhorn(
        matrix_in,
        matrix_out,
        mu1_out,
        mu2_out,
        *,
        order: int = 10,
        lr: float = 0.5,
        eps: float = 1e-3,
        block_dim: int = block_dim,
        stream_ptr: Optional[int] = None,
    ):
        N, K, L = matrix_in.shape
        _validate_io(matrix_in, matrix_out, mu1_out, mu2_out, K, L)

        inv_k = 1.0 / K
        inv_l = 1.0 / L
        inv_k1 = 1.0 / (K - 1) if K > 1 else 1.0
        inv_l1 = 1.0 / (L - 1) if L > 1 else 1.0

        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        lib.call_sinkhorn_kernel(
            block_dim,
            stream_ptr,
            _torch_to_ctypes(matrix_in),
            _torch_to_ctypes(matrix_out),
            _torch_to_ctypes(mu1_out),
            _torch_to_ctypes(mu2_out),
            int(N),
            int(K),
            int(L),
            int(order),
            float(lr),
            float(eps),
            float(inv_k),
            float(inv_l),
            float(inv_k1),
            float(inv_l1),
        )

    sinkhorn.block_dim = block_dim
    return sinkhorn


def jit_compile_pto(
    builder_py: Path | str = THIS_DIR / "sinkhorn_builder.py",
    *,
    verbose: bool = True,
    so_dir: Path | str = DEFAULT_SO_DIR,
    block_dim: int = BLOCK_DIM,
    force: bool = False,
) -> Callable:
    """One-shot: build PTODSL kernel + return loaded callable."""
    so = compile_pto_lib(builder_py, so_dir=so_dir, verbose=verbose, force=force)
    return load_lib(so, block_dim=block_dim)


def jit_compile_reference(
    reference_cpp: Path | str = THIS_DIR / "reference.cpp",
    *,
    verbose: bool = True,
    so_dir: Path | str = DEFAULT_SO_DIR,
    block_dim: int = BLOCK_DIM,
    force: bool = False,
) -> Callable:
    """One-shot: build reference.cpp + return loaded callable."""
    so = compile_reference_lib(
        reference_cpp, so_dir=so_dir, verbose=verbose, force=force
    )
    return load_lib(so, block_dim=block_dim)


def clean_cache(so_dir: Path | str = DEFAULT_SO_DIR) -> None:
    """Remove all cached build artefacts."""
    so_dir = Path(so_dir)
    if so_dir.exists():
        shutil.rmtree(so_dir)
