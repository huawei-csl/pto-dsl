#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# --------------------------------------------------------------------------------

"""JIT compile split-pipe fa_performance_kernel.cpp + host dispatch (LaunchTFA)."""

import ctypes
import os
import subprocess
from pathlib import Path
from typing import List, Optional

import torch

ASCEND_TOOLKIT_HOME = os.environ["ASCEND_TOOLKIT_HOME"]
_SPLITPIPE_ROOT = Path(os.environ.get("PTO_ISA_SPLITPIPE", "/workdir/pto-isa-splitpipe")).resolve()
_SPLIT_PIPE_DIR = Path(__file__).resolve().parent

_CV_FIFO_SIZE = 8
_CUBE_S0 = 128
_CUBE_S1 = 128
_SUPPORTED_TILE_S1 = (256, 512, 1024)
_DEFAULT_TILE_S1 = 256


def torch_to_ctypes(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _npu_arch_flag() -> str:
    return os.environ.get("NPU_ARCH", "dav-2201").strip()


def compile_flash(
    kernel_cpp: str,
    verbose: bool = False,
    timeout: int = 600,
    extra_sources: Optional[List[str]] = None,
    splitpipe_root: Optional[Path] = None,
    output_lib: Optional[str] = None,
) -> str:
    lib_path = output_lib or str(_SPLIT_PIPE_DIR / "flash_jit.so")
    root = splitpipe_root or _SPLITPIPE_ROOT

    includes = [
        f"-I{root}/include",
        f"-I{root}/kernels/manual/common/flash_atten",
        f"-I{_SPLIT_PIPE_DIR}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
    ]
    pto_extra = os.environ.get("PTO_LIB_PATH")
    if pto_extra:
        includes.insert(0, f"-I{pto_extra}/include")

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        f"--npu-arch={_npu_arch_flag()}",
        "-O2",
        "-std=c++17",
        "-Wno-ignored-attributes",
        *includes,
    ]

    sources = [kernel_cpp]
    if extra_sources:
        sources.extend(extra_sources)

    cmd = ["bisheng", *flags, *sources, "-o", lib_path]
    if verbose:
        print("compile command:\n", " ".join(cmd))

    subprocess.run(cmd, check=True, timeout=timeout)

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def load_flash_lib(lib_path: str, check_type: bool = True):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    if check_type:
        lib.call_kernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.call_kernel.restype = None

    _ws: dict = {}

    def _alloc_workspace(s0: int, head: int, tile_s1: int, device):
        # Intermediate GM layouts depend on S0/HEAD/tile_s1 only (see demos jit_util_flash).
        shape = (s0, head, tile_s1, str(device))
        if _ws.get("_shape") == shape:
            return

        torch.npu.synchronize()

        if tile_s1 not in _SUPPORTED_TILE_S1:
            raise ValueError(f"tile_s1 must be one of {_SUPPORTED_TILE_S1}, got {tile_s1}")

        num_s0_blocks = s0 // _CUBE_S0
        slots = num_s0_blocks * _CV_FIFO_SIZE

        _ws.clear()
        _ws["_shape"] = shape
        _ws["o_out"] = torch.empty((s0, head), device=device, dtype=torch.float32)

        _ws["qk_tile_fifo"] = torch.empty(
            (slots, _CUBE_S0, tile_s1), device=device, dtype=torch.float32
        )
        _ws["p_tile_fifo"] = torch.empty(
            (slots, _CUBE_S0, tile_s1), device=device, dtype=torch.float16
        )
        _ws["exp_max_ififo"] = torch.empty(
            (slots, _CUBE_S0), device=device, dtype=torch.float32
        )
        _ws["pv_tile_fifo"] = torch.empty(
            (slots, _CUBE_S0, head), device=device, dtype=torch.float32
        )

        # Match demos/torch_jit/flash_atten/jit_util_flash.py layout for LaunchTFA intermediates.
        _ws["global_sum_out"] = torch.empty(
            (num_s0_blocks, s0), device=device, dtype=torch.float32
        )
        _ws["exp_max_out"] = torch.empty(
            (num_s0_blocks, s0), device=device, dtype=torch.float32
        )
        _ws["o_parts_out"] = torch.empty(
            (num_s0_blocks, s0, head), device=device, dtype=torch.float32
        )

    default_causal = False
    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def flash(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        stream_ptr=default_stream_ptr,
        is_causal=default_causal,
        tile_s1: int = _DEFAULT_TILE_S1,
    ) -> torch.Tensor:
        s1 = k.shape[0]
        if s1 % tile_s1 != 0:
            raise ValueError(f"S1={s1} must be divisible by tile_s1={tile_s1}")
        _alloc_workspace(q.shape[0], q.shape[1], tile_s1, q.device)

        lib.call_kernel(
            stream_ptr,
            q.shape[1],
            q.shape[0],
            k.shape[0],
            tile_s1,
            is_causal,
            torch_to_ctypes(q),
            torch_to_ctypes(k),
            torch_to_ctypes(v),
            torch_to_ctypes(_ws["o_out"]),
            torch_to_ctypes(_ws["qk_tile_fifo"]),
            torch_to_ctypes(_ws["p_tile_fifo"]),
            torch_to_ctypes(_ws["exp_max_ififo"]),
            torch_to_ctypes(_ws["pv_tile_fifo"]),
            torch_to_ctypes(_ws["global_sum_out"]),
            torch_to_ctypes(_ws["exp_max_out"]),
            torch_to_ctypes(_ws["o_parts_out"]),
        )
        return _ws["o_out"]

    return flash


def jit_compile_flash(
    verbose: bool = False,
    clean_up: bool = True,
    kernel_cpp: Optional[str] = None,
    splitpipe_root: Optional[Path] = None,
):
    root = splitpipe_root or _SPLITPIPE_ROOT
    kcpp = kernel_cpp or str(
        root / "kernels/manual/common/flash_atten/fa_performance_kernel.cpp"
    )
    dispatch = str(_SPLIT_PIPE_DIR / "call_kernel_dispatch.cpp")
    lib_path = compile_flash(
        kcpp,
        verbose=verbose,
        extra_sources=[dispatch],
        splitpipe_root=root,
        output_lib=str(_SPLIT_PIPE_DIR / "flash_jit.so"),
    )
    func = load_flash_lib(lib_path)

    if clean_up:
        try:
            os.remove(lib_path)
        except OSError:
            pass

    return func
