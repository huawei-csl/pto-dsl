import ctypes
import os
import pathlib
import subprocess
from dataclasses import dataclass
from functools import update_wrapper

from mlir.dialects import pto as _pto
from mlir.ir import Context, Location

from .._constexpr import (
    _MISSING,
    analyze_signature,
    bind_kernel_arguments,
    normalize_constexpr_bindings,
    specialization_suffix,
)
from .ir import _build_ir_module


def _type_repr(type_obj):
    return str(type_obj).replace(" ", "").lower()


def _is_ptr_type(type_obj):
    return "ptr" in _type_repr(type_obj)


def _ptr_elem_cpp_type(type_obj):
    type_repr = _type_repr(type_obj)
    if "e8m0" in type_repr:
        return "float8_e8m0_t"
    if "e4m3" in type_repr:
        return "float8_e4m3_t"
    if "e5m2" in type_repr:
        return "float8_e5m2_t"
    if "f32" in type_repr:
        return "float"
    if "f16" in type_repr:
        return "__fp16"
    if "bf16" in type_repr:
        return "__bf16"
    if "i8" in type_repr:
        return "int8_t"
    if "u8" in type_repr:
        return "uint8_t"
    if "i16" in type_repr:
        return "int16_t"
    if "u16" in type_repr:
        return "uint16_t"
    if "i32" in type_repr:
        return "int32_t"
    if "u32" in type_repr:
        return "uint32_t"
    if "i64" in type_repr:
        return "int64_t"
    if "u64" in type_repr:
        return "uint64_t"
    return "float"


def _scalar_cpp_type(type_obj):
    type_repr = _type_repr(type_obj)
    if "i32" in type_repr:
        return "int32_t"
    if "i64" in type_repr or "index" in type_repr:
        return "int64_t"
    if "e8m0" in type_repr or "e4m3" in type_repr or "e5m2" in type_repr:
        return "uint8_t"
    if "f32" in type_repr:
        return "float"
    if "f16" in type_repr:
        return "__fp16"
    return "int32_t"


def _scalar_ctype(type_obj):
    type_repr = _type_repr(type_obj)
    if "i64" in type_repr or "index" in type_repr:
        return ctypes.c_int64
    if "e8m0" in type_repr or "e4m3" in type_repr or "e5m2" in type_repr:
        return ctypes.c_uint8
    if "f32" in type_repr:
        return ctypes.c_float
    if "f16" in type_repr:
        return ctypes.c_uint16
    return ctypes.c_int32


def _normalize_stream_ptr(stream_ptr):
    if isinstance(stream_ptr, ctypes.c_void_p):
        return stream_ptr
    if isinstance(stream_ptr, int):
        return ctypes.c_void_p(stream_ptr)
    if hasattr(stream_ptr, "value"):
        return ctypes.c_void_p(int(stream_ptr.value))
    return stream_ptr


@dataclass
class _CompiledSpecialization:
    arg_types: list[object]
    lib: object
    lib_path: pathlib.Path
    output_dir: pathlib.Path


class JitWrapper:
    def __init__(
        self,
        fn,
        *,
        meta_data,
        output_dir=None,
        block_dim=20,
        enable_insert_sync=True,
        npu_arch="dav-2201",
    ):
        self._fn = fn
        self._meta_data = meta_data
        self._analysis = analyze_signature(fn)
        self._sig = self._analysis.signature
        self._runtime_params = list(self._analysis.runtime_params)
        self._arg_types = None
        self._output_dir = (
            pathlib.Path(output_dir)
            if output_dir
            else pathlib.Path.cwd() / ".ptodsl_jit" / fn.__name__
        )
        self._block_dim = block_dim
        self._enable_insert_sync = enable_insert_sync
        self._npu_arch = npu_arch
        self._compiled = False
        self._lib = None
        self._compiled_specializations = {}
        self._lib_path = self._output_dir / "kernel.so"
        update_wrapper(self, fn)

    def _artifact_paths(self, output_dir):
        pto_path = output_dir / "kernel.pto"
        cpp_path = output_dir / "kernel.cpp"
        caller_path = output_dir / "caller.cpp"
        lib_path = output_dir / "kernel.so"
        return pto_path, cpp_path, caller_path, lib_path

    def _generate_caller_cpp(self, kernel_cpp_name, runtime_params=None):
        params = list(
            self._runtime_params if runtime_params is None else runtime_params
        )
        cpp_args = []
        launch_args = []
        for param, arg_type in zip(params, self._arg_types):
            if _is_ptr_type(arg_type):
                cpp_args.append(f"uint8_t *{param.name}")
                launch_args.append(f"({_ptr_elem_cpp_type(arg_type)} *){param.name}")
            else:
                cpp_t = _scalar_cpp_type(arg_type)
                cpp_args.append(f"{cpp_t} {param.name}")
                launch_args.append(param.name)

        wrapper_sig = ", ".join(["uint32_t blockDim", "void *stream"] + cpp_args)
        kernel_call = ", ".join(launch_args)
        return (
            f'#include "{kernel_cpp_name}"\n'
            f"#include <cstdint>\n\n"
            f'extern "C" void call_kernel({wrapper_sig})\n'
            "{\n"
            f"    {self._fn.__name__}<<<blockDim, nullptr, stream>>>({kernel_call});\n"
            "}\n"
        )

    def _compile_shared_library(self, caller_cpp_path, lib_path, *, cwd):
        toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME")
        if not toolkit_home:
            raise RuntimeError(
                "ASCEND_TOOLKIT_HOME is required to compile generated caller.cpp."
            )
        cmd = [
            "bisheng",
            f"-I{toolkit_home}/include",
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
            f"--npu-arch={self._npu_arch}",
            "-DMEMORY_BASE",  # TODO: add switch for A5
            "-std=gnu++17",
            str(caller_cpp_path),
            "-o",
            str(lib_path),
        ]
        subprocess.run(cmd, check=True, cwd=str(cwd))

    def _resolve_runtime_arg_types(self, constexpr_bindings):
        from .ir import _resolve_arg_types, _resolve_meta

        with Context() as ctx, Location.unknown():
            _pto.register_dialect(ctx, load=True)
            meta_map = _resolve_meta(self._meta_data, constexpr_bindings)
            return _resolve_arg_types(self._runtime_params, meta_map)

    def _specialization_output_dir(self, constexpr_bindings):
        if not self._analysis.has_constexpr_params:
            return self._output_dir
        return self._output_dir / f"spec_{specialization_suffix(constexpr_bindings)}"

    def _build(self, constexpr_bindings):
        output_dir = self._specialization_output_dir(constexpr_bindings)
        output_dir.mkdir(parents=True, exist_ok=True)
        pto_path, cpp_path, caller_path, lib_path = self._artifact_paths(output_dir)
        self._arg_types = self._resolve_runtime_arg_types(constexpr_bindings)

        ir_module = _build_ir_module(
            self._fn, self._analysis, self._meta_data, constexpr_bindings
        )
        pto_path.write_text(f"{ir_module}\n", encoding="utf-8")

        ptoas_cmd = ["ptoas"]
        if self._enable_insert_sync:
            ptoas_cmd.append("--enable-insert-sync")
        ptoas_cmd += [str(pto_path), "-o", str(cpp_path)]
        subprocess.run(ptoas_cmd, check=True, cwd=str(output_dir))

        caller_path.write_text(
            self._generate_caller_cpp(cpp_path.name), encoding="utf-8"
        )
        self._compile_shared_library(caller_path, lib_path, cwd=output_dir)

        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.call_kernel.argtypes = [ctypes.c_uint32, ctypes.c_void_p] + [
            ctypes.c_void_p if _is_ptr_type(arg_type) else _scalar_ctype(arg_type)
            for arg_type in self._arg_types
        ]
        self._compiled = True
        self._lib_path = lib_path
        return _CompiledSpecialization(
            arg_types=list(self._arg_types),
            lib=self._lib,
            lib_path=lib_path,
            output_dir=output_dir,
        )

    def _convert_ptr(self, value):
        if isinstance(value, ctypes.c_void_p):
            return value
        if hasattr(value, "data_ptr"):
            return ctypes.c_void_p(value.data_ptr())
        if isinstance(value, int):
            return ctypes.c_void_p(value)
        raise TypeError(f"Pointer-like argument expected, got {type(value)!r}.")

    def _prepare_call_args(self, runtime_values, arg_types):
        converted = []
        for param, value, arg_type in zip(
            self._runtime_params, runtime_values, arg_types
        ):
            if value is _MISSING:
                raise TypeError(f"Missing required argument '{param.name}'.")
            if _is_ptr_type(arg_type):
                converted.append(self._convert_ptr(value))
            else:
                converted.append(value)
        return converted

    def __call__(self, *args, stream_ptr=None, **kwargs):
        bound = bind_kernel_arguments(self._analysis, *args, **kwargs)
        constexpr_key = normalize_constexpr_bindings(bound.constexpr_arguments)

        specialization = self._compiled_specializations.get(constexpr_key)
        if specialization is None:
            specialization = self._build(bound.constexpr_arguments)
            self._compiled_specializations[constexpr_key] = specialization

        if stream_ptr is None:
            import torch

            stream_ptr = torch.npu.current_stream()._as_parameter_

        call_args = self._prepare_call_args(
            bound.runtime_arguments, specialization.arg_types
        )
        specialization.lib.call_kernel(
            ctypes.c_uint32(self._block_dim),
            _normalize_stream_ptr(stream_ptr),
            *call_args,
        )
        return None

    def set_block_dim(self, block_dim):
        if not isinstance(block_dim, int) or block_dim <= 0:
            raise ValueError("`block_dim` must be a positive integer.")
        self._block_dim = block_dim
        return self

    @property
    def library_path(self):
        return str(self._lib_path)

    @property
    def output_dir(self):
        return str(self._output_dir)


def jit(
    *,
    meta_data,
    output_dir=None,
    block_dim=1,
    enable_insert_sync=True,
    npu_arch="dav-2201",
):
    def decorator(fn):
        return JitWrapper(
            fn,
            meta_data=meta_data,
            output_dir=output_dir,
            block_dim=block_dim,
            enable_insert_sync=enable_insert_sync,
            npu_arch=npu_arch,
        )

    return decorator


__all__ = ["JitWrapper", "jit"]
