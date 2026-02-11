import inspect
import ctypes
import os
import pathlib
import subprocess
from functools import update_wrapper

from mlir.dialects import func, pto
from mlir.ir import Context, InsertionPoint, Location, Module

from .language import wrap_value


def _resolve_meta(meta_fn):
    values = meta_fn()
    if not isinstance(values, dict):
        raise ValueError("`meta_data()` must return a dict of named symbols to MLIR/PTO types.")
    return dict(values)


def _resolve_arg_types(signature, meta_map):
    arg_types = []
    for param in signature.parameters.values():
        annot = param.annotation
        if isinstance(annot, str):
            if annot not in meta_map:
                raise ValueError(f"Unknown annotation '{annot}'.")
            arg_types.append(meta_map[annot])
        elif annot is inspect._empty:
            raise ValueError(f"Missing annotation for argument '{param.name}'.")
        else:
            arg_types.append(annot)
    return arg_types


def _resolve_ret_types(signature, meta_map):
    ret_annot = signature.return_annotation
    if ret_annot in (inspect._empty, None):
        return []
    if isinstance(ret_annot, str):
        if ret_annot not in meta_map:
            raise ValueError(f"Unknown return annotation '{ret_annot}'.")
        return [meta_map[ret_annot]]
    if isinstance(ret_annot, (list, tuple)):
        out = []
        for elem in ret_annot:
            if isinstance(elem, str):
                out.append(meta_map[elem])
            else:
                out.append(elem)
        return out
    return [ret_annot]


def _has_func_return(block):
    last_name = None
    for op in block.operations:
        last_name = op.operation.name
    return last_name == "func.return"


def _inject_globals(fn, values):
    old = {}
    for name, value in values.items():
        old[name] = fn.__globals__.get(name, None)
        fn.__globals__[name] = value
    return old


def _restore_globals(fn, old, injected_names):
    for name in injected_names:
        if old[name] is None and name in fn.__globals__:
            del fn.__globals__[name]
        else:
            fn.__globals__[name] = old[name]


def to_ir_module(*, meta_data):
    def decorator(fn):
        sig = inspect.signature(fn)

        with Context() as ctx, Location.unknown():
            pto.register_dialect(ctx, load=True)
            meta_map = _resolve_meta(meta_data)
            arg_types = _resolve_arg_types(sig, meta_map)
            ret_types = _resolve_ret_types(sig, meta_map)
            module = Module.create()
            fn_ty = func.FunctionType.get(arg_types, ret_types)

            with InsertionPoint(module.body):
                ir_func = func.FuncOp(fn.__name__, fn_ty)
                entry = ir_func.add_entry_block()

            with InsertionPoint(entry):
                wrapped_args = [wrap_value(arg) for arg in entry.arguments]
                injected = set(meta_map.keys())
                old_globals = _inject_globals(fn, meta_map)
                try:
                    fn(*wrapped_args)
                finally:
                    _restore_globals(fn, old_globals, injected)

                if not ret_types and not _has_func_return(entry):
                    func.ReturnOp([])

            module.operation.verify()
            return module

    return decorator


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
    if "f32" in type_repr:
        return "float"
    if "f16" in type_repr:
        return "__fp16"
    return "int32_t"


def _scalar_ctype(type_obj):
    type_repr = _type_repr(type_obj)
    if "i64" in type_repr or "index" in type_repr:
        return ctypes.c_int64
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
        self._sig = inspect.signature(fn)
        self._arg_types = None
        self._output_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path.cwd() / ".ptodsl_jit" / fn.__name__
        self._block_dim = block_dim
        self._enable_insert_sync = enable_insert_sync
        self._npu_arch = npu_arch
        self._compiled = False
        self._lib = None
        self._lib_path = self._output_dir / "kernel.so"
        update_wrapper(self, fn)

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
            if _is_ptr_type(arg_type):
                cpp_args.append(f"uint8_t *{param.name}")
                launch_args.append(f"({ _ptr_elem_cpp_type(arg_type) } *){param.name}")
            else:
                cpp_t = _scalar_cpp_type(arg_type)
                cpp_args.append(f"{cpp_t} {param.name}")
                launch_args.append(param.name)

        wrapper_sig = ", ".join(["void *stream"] + cpp_args)
        kernel_call = ", ".join(launch_args)
        return (
            f'#include "{kernel_cpp_name}"\n'
            f"#include <cstdint>\n\n"
            f'extern "C" void call_kernel({wrapper_sig})\n'
            "{\n"
            f"    {self._fn.__name__}<<<{self._block_dim}, nullptr, stream>>>({kernel_call});\n"
            "}\n"
        )

    def _compile_shared_library(self, caller_cpp_path, lib_path):
        toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME")
        if not toolkit_home:
            raise RuntimeError("ASCEND_TOOLKIT_HOME is required to compile generated caller.cpp.")
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
            "-DMEMORY_BASE",
            "-std=gnu++17",
            str(caller_cpp_path),
            "-o",
            str(lib_path),
        ]
        subprocess.run(cmd, check=True, cwd=str(self._output_dir))

    def _resolve_runtime_arg_types(self):
        with Context() as ctx, Location.unknown():
            pto.register_dialect(ctx, load=True)
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

        caller_path.write_text(self._generate_caller_cpp(cpp_path.name), encoding="utf-8")
        self._compile_shared_library(caller_path, lib_path)

        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.call_kernel.argtypes = [ctypes.c_void_p] + [
            ctypes.c_void_p if _is_ptr_type(arg_type) else _scalar_ctype(arg_type)
            for arg_type in self._arg_types
        ]
        self._compiled = True

    def _convert_ptr(self, value):
        if isinstance(value, ctypes.c_void_p):
            return value
        if hasattr(value, "data_ptr"):
            return ctypes.c_void_p(value.data_ptr())
        if isinstance(value, int):
            return ctypes.c_void_p(value)
        raise TypeError(f"Pointer-like argument expected, got {type(value)!r}.")

    def _default_scalar_value(self, param_name):
        lower_name = param_name.lower()
        if "vrow" in lower_name or "valid_row" in lower_name:
            return 32
        if "vcol" in lower_name or "valid_col" in lower_name:
            return 32
        return 0

    def _prepare_call_args(self, args):
        params = list(self._sig.parameters.values())
        if len(args) > len(params):
            raise TypeError(f"Expected at most {len(params)} arguments, got {len(args)}.")

        filled_args = list(args)
        for idx in range(len(args), len(params)):
            param = params[idx]
            if param.default is not inspect._empty:
                filled_args.append(param.default)
                continue
            arg_type = self._arg_types[idx]
            if _is_ptr_type(arg_type):
                raise TypeError(f"Missing required pointer argument '{param.name}'.")
            filled_args.append(self._default_scalar_value(param.name))

        converted = []
        for value, arg_type in zip(filled_args, self._arg_types):
            if _is_ptr_type(arg_type):
                converted.append(self._convert_ptr(value))
            else:
                converted.append(value)
        return converted

    def __call__(self, *args, stream_ptr=None):
        if not self._compiled:
            self._build()

        if stream_ptr is None:
            try:
                import torch

                stream_ptr = torch.npu.current_stream()._as_parameter_
            except Exception as exc:
                raise RuntimeError(
                    "stream_ptr is not provided and torch.npu current stream could not be resolved."
                ) from exc

        call_args = self._prepare_call_args(args)
        self._lib.call_kernel(_normalize_stream_ptr(stream_ptr), *call_args)
        return None

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
    block_dim=20,
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


__all__ = ["JitWrapper", "jit", "to_ir_module"]
