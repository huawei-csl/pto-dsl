import ctypes
import inspect
import os
import pathlib
import subprocess
from functools import update_wrapper, wraps

from mlir.dialects import pto as _pto
from mlir.ir import Context, Location

from ..utils.npu_info import get_num_cube_cores
from .ir import to_ir_module


def _type_repr(type_obj):
    return str(type_obj).replace(" ", "").lower()


def _is_ptr_type(type_obj):
    return "ptr" in _type_repr(type_obj)


def _ptr_elem_cpp_type(type_obj):
    type_repr = _type_repr(type_obj)
    if "f32" in type_repr:
        return "float"
    if "f16" in type_repr:
        return "__fp16"
    if "bf16" in type_repr:
        return "__bf16"
    if "ui8" in type_repr or "u8" in type_repr:
        return "uint8_t"
    if "ui16" in type_repr or "u16" in type_repr:
        return "uint16_t"
    if "ui32" in type_repr or "u32" in type_repr:
        return "uint32_t"
    if "ui64" in type_repr or "u64" in type_repr:
        return "uint64_t"
    if "i8" in type_repr:
        return "int8_t"
    if "i16" in type_repr:
        return "int16_t"
    if "i32" in type_repr:
        return "int32_t"
    if "i64" in type_repr:
        return "int64_t"
    return "float"


def _scalar_cpp_type(type_obj):
    type_repr = _type_repr(type_obj)
    if "ui32" in type_repr or "u32" in type_repr:
        return "uint32_t"
    if "ui64" in type_repr or "u64" in type_repr:
        return "uint64_t"
    if "i32" in type_repr:
        return "int32_t"
    if "i64" in type_repr or "index" in type_repr:
        return "int64_t"
    if "f32" in type_repr:
        return "float"
    if "f16" in type_repr:
        return "__fp16"
    return "int32_t"


def _scalar_ctype(type_obj):
    type_repr = _type_repr(type_obj)
    if "ui64" in type_repr or "u64" in type_repr:
        return ctypes.c_uint64
    if "i64" in type_repr or "index" in type_repr:
        return ctypes.c_int64
    if "f32" in type_repr:
        return ctypes.c_float
    if "f16" in type_repr:
        return ctypes.c_uint16
    if "ui32" in type_repr or "u32" in type_repr:
        return ctypes.c_uint32
    return ctypes.c_int32


def _normalize_stream_ptr(stream_ptr):
    if isinstance(stream_ptr, ctypes.c_void_p):
        return stream_ptr
    if isinstance(stream_ptr, int):
        return ctypes.c_void_p(stream_ptr)
    if hasattr(stream_ptr, "value"):
        return ctypes.c_void_p(int(stream_ptr.value))
    return stream_ptr


class JitWrapper:
    def __init__(
        self,
        fn,
        *,
        meta_data,
        output_dir=None,
        block_dim=None,
        enable_insert_sync=True,
        init_ffts=None,
        npu_arch="dav-2201",
    ):
        self._fn = fn
        self._orig_sig = inspect.signature(fn)
        self._sig = self._orig_sig
        self._meta_data = meta_data
        self._arg_types = None
        self._output_dir = (
            pathlib.Path(output_dir)
            if output_dir
            else pathlib.Path.cwd() / ".ptodsl_jit" / fn.__name__
        )
        self._block_dim = block_dim if block_dim is not None else get_num_cube_cores()
        self._enable_insert_sync = enable_insert_sync
        self._init_ffts = init_ffts
        self._npu_arch = npu_arch
        self._compiled = False
        self._lib = None
        self._lib_path = self._output_dir / "kernel.so"
        update_wrapper(self, fn)

        if self._init_ffts is not None:
            original_fn = self._fn

            @wraps(original_fn)
            def wrapper(*args, **kwargs):
                # Automatically emit the MLIR operation before tracing the rest of the kernel
                from ..api import pto as pto_api

                pto_api.set_ffts(args[-1])
                return original_fn(*args[:-1], **kwargs)

            new_params = list(self._sig.parameters.values())
            new_params.append(
                inspect.Parameter(
                    self._init_ffts,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation="ffts_type",
                )
            )
            self._sig = self._sig.replace(parameters=new_params)
            wrapper.__signature__ = self._sig
            self._fn = wrapper

    def _artifact_paths(self):
        pto_path = self._output_dir / "kernel.pto"
        cpp_path = self._output_dir / "kernel.cpp"
        caller_path = self._output_dir / "caller.cpp"
        return pto_path, cpp_path, caller_path, self._lib_path

    def _generate_caller_cpp(self, kernel_cpp_name):
        params = list(self._sig.parameters.values())
        cpp_args = []
        launch_args = []
        for param, arg_type in zip(params, self._arg_types):
            if param.name == self._init_ffts:
                launch_args.append(f"reinterpret_cast<uint64_t *>(fftsAddr)")
            else:
                if _is_ptr_type(arg_type):
                    cpp_args.append(f"uint8_t *{param.name}")
                    launch_args.append(
                        f"({_ptr_elem_cpp_type(arg_type)} *){param.name}"
                    )
                else:
                    cpp_t = _scalar_cpp_type(arg_type)
                    cpp_args.append(f"{cpp_t} {param.name}")
                    launch_args.append(param.name)

        wrapper_sig = ", ".join(["uint32_t blockDim", "void *stream"] + cpp_args)
        kernel_call = ", ".join(launch_args)

        ffts_init_code = ""
        if self._init_ffts is not None:
            ffts_init_code = (
                "    void *fftsAddr = nullptr;\n"
                "    uint32_t fftsLen = 0;\n"
                "    (void)rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&fftsAddr), &fftsLen);\n"
            )

        return (
            f'#include "{kernel_cpp_name}"\n'
            f"#include <cstdint>\n"
            f'#include "runtime/rt.h"\n\n'
            f'extern "C" void call_kernel({wrapper_sig})\n'
            "{\n"
            f"{ffts_init_code}"
            f"    {self.__name__}<<<blockDim, nullptr, stream>>>({kernel_call});\n"
            "}\n"
        )

    def _compile_shared_library(self, caller_cpp_path, lib_path):
        # CANN 8.5 headers don't have CompactMode, need latest pto-isa source
        pto_isa = os.environ.get("PTO_LIB_PATH", "/sources/pto-isa")
        if not pto_isa:
            raise RuntimeError(
                "PTO_LIB_PATH is required to compile generated caller.cpp."
            )
        ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME")
        cmd = [
            "bisheng",
            f"-I{pto_isa}/include",
            f"-I{ascend_home}/include",
            f"-I{ascend_home}/pkg_inc",
            f"-I{ascend_home}/pkg_inc/runtime",
            f"-I{ascend_home}/pkg_inc/profiling",
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
        try:
            subprocess.run(cmd, check=True, cwd=str(self._output_dir))
        except Exception as e:
            output = (
                e.stdout.decode("utf-8", errors="replace")
                if hasattr(e, "stdout") and e.stdout
                else ""
            )
            raise RuntimeError(
                f"Compile failed with exit code {e.returncode}:\n{output}"
            ) from e

    def _resolve_runtime_arg_types(self):
        from .ir import _resolve_arg_types, _resolve_meta

        with Context() as ctx, Location.unknown():
            _pto.register_dialect(ctx, load=True)
            meta_map = _resolve_meta(self._meta_data)
            return _resolve_arg_types(self._sig, meta_map)

    def _build(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        pto_path, cpp_path, caller_path, lib_path = self._artifact_paths()
        self._arg_types = self._resolve_runtime_arg_types()

        ir_module = to_ir_module(meta_data=self._meta_data)(self._fn)
        pto_path.write_text(f"{ir_module}\n", encoding="utf-8")

        ptoas_cmd = ["ptoas"]
        if self._enable_insert_sync:
            ptoas_cmd.append("--enable-insert-sync")
        ptoas_cmd += [str(pto_path), "-o", str(cpp_path)]
        subprocess.run(ptoas_cmd, check=True, cwd=str(self._output_dir))

        caller_path.write_text(
            self._generate_caller_cpp(cpp_path.name), encoding="utf-8"
        )
        self._compile_shared_library(caller_path, lib_path)

        self._lib = ctypes.CDLL(str(lib_path))
        argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        for param, arg_type in zip(self._sig.parameters.values(), self._arg_types):
            if self._init_ffts is not None and param.name == self._init_ffts:
                continue
            if _is_ptr_type(arg_type):
                argtypes.append(ctypes.c_void_p)
            else:
                argtypes.append(_scalar_ctype(arg_type))

        self._lib.call_kernel.argtypes = argtypes
        self._compiled = True

    def _convert_ptr(self, value):
        if isinstance(value, ctypes.c_void_p):
            return value
        if hasattr(value, "data_ptr"):
            return ctypes.c_void_p(value.data_ptr())
        if isinstance(value, int):
            return ctypes.c_void_p(value)
        raise TypeError(f"Pointer-like argument expected, got {type(value)!r}.")

    def _prepare_call_args(self, args):
        params = list(self._sig.parameters.values())
        orig_params = [
            p for p in params if self._init_ffts is None or p.name != self._init_ffts
        ]
        orig_arg_types = [
            t
            for p, t in zip(params, self._arg_types)
            if self._init_ffts is None or p.name != self._init_ffts
        ]

        if len(args) > len(orig_params):
            raise TypeError(
                f"Expected at most {len(orig_params)} arguments, got {len(args)}."
            )

        filled_args = list(args)
        for idx in range(len(filled_args), len(orig_params)):
            param = orig_params[idx]
            if param.default is not inspect._empty:
                filled_args.append(param.default)
                continue
            arg_type = orig_arg_types[idx]
            if _is_ptr_type(arg_type):
                raise TypeError(f"Missing required pointer argument '{param.name}'.")

        converted = []
        for value, arg_type in zip(filled_args, orig_arg_types):
            if _is_ptr_type(arg_type):
                converted.append(self._convert_ptr(value))
            else:
                converted.append(value)
        return converted

    # TODO: also allow taking named `kwargs`
    def __call__(self, *args, stream_ptr=None):
        if not self._compiled:
            self._build()

        if stream_ptr is None:
            import torch

            stream_ptr = torch.npu.current_stream()._as_parameter_

        call_args = self._prepare_call_args(args)
        self._lib.call_kernel(
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
    init_ffts=None,
    npu_arch="dav-2201",
):
    def decorator(fn):
        return JitWrapper(
            fn,
            meta_data=meta_data,
            output_dir=output_dir,
            block_dim=block_dim,
            enable_insert_sync=enable_insert_sync,
            init_ffts=init_ffts,
            npu_arch=npu_arch,
        )

    return decorator


__all__ = ["JitWrapper", "jit"]
