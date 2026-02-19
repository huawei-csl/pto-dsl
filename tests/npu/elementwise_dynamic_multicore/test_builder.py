import sys
import os
import ctypes
import subprocess

import pytest
import torch
import torch_npu

torch.manual_seed(0)

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = "npu:5"

BINARY_OPS = [
    ("add", lambda x, y: x + y),
    ("sub", lambda x, y: x - y),
    ("mul", lambda x, y: x * y),
    ("div", lambda x, y: x / y),
    #("or", lambda x, y: x | y), #TODO add back bitwise or test after fixing int16 support in the builder
]

DTYPES = ["float32", "float16", "int16"]

# ops that only make sense for floating-point dtypes
_FLOAT_ONLY_OPS = {"add", "sub", "mul", "div"}
# ops that only make sense for int16
_INT16_ONLY_OPS = {"or"}


TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int32": torch.int32,
    "int16": torch.int16,
}


def _make_params():
    for op_name, ref_fn in BINARY_OPS:
        for dtype in DTYPES:
            marks = []
            if op_name in _FLOAT_ONLY_OPS and dtype in ("int32", "int16"):
                marks.append(
                    pytest.mark.skip(reason=f"{op_name} not supported for {dtype}")
                )
            if op_name in _INT16_ONLY_OPS and dtype != "int16":
                marks.append(
                    pytest.mark.skip(reason=f"{op_name} only supported for int16")
                )
            yield pytest.param(
                (op_name, ref_fn, dtype), id=f"{op_name}-{dtype}", marks=marks
            )


_PARAMS = list(_make_params())


@pytest.fixture(scope="session", params=_PARAMS)
def compiled_lib(request):
    op_name, ref_fn, dtype = request.param
    subprocess.check_call(
        ["bash", os.path.join(_DIR, "compile.sh"), op_name, dtype],
        cwd=_DIR,
    )
    yield {
        "op_name": op_name,
        "ref_fn": ref_fn,
        "dtype": dtype,
        "lib": ctypes.CDLL(_lib_path(op_name, dtype)),
    }
    os.remove(_lib_path(op_name, dtype))


def _make_input(shape, device, dtype):
    if dtype.is_floating_point:
        return torch.rand(shape, device=device, dtype=dtype) + 0.1
    else:
        return torch.randint(
            0,
            1000,
            (shape,) if isinstance(shape, int) else shape,
            device=device,
            dtype=dtype,
        )


def _ctypes_ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def _lib_to_func_binary_1d(lib):
    def fn(x, y, z):
        stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel_1d(
            stream_ptr, _ctypes_ptr(x), _ctypes_ptr(y), _ctypes_ptr(z), x.numel()
        )

    return fn


def _lib_to_func_binary_2d(lib):
    def fn(x, y, z):
        stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel_2d(
            stream_ptr,
            _ctypes_ptr(x),
            _ctypes_ptr(y),
            _ctypes_ptr(z),
            x.size(0),
            x.size(1),
        )

    return fn


def _lib_path(op_name, dtype):
    return os.path.join(_DIR, f"{op_name}_{dtype}_lib.so")


# Compilation happens inside the fixture; this test just confirms the .so landed.


def test_build_binary_kernels(compiled_lib):
    assert os.path.exists(_lib_path(compiled_lib["op_name"], compiled_lib["dtype"]))


# The fixture guarantees the lib is compiled before either test runs.


@pytest.mark.require_npu
def test_binary_1d_precision(compiled_lib):
    torch.npu.set_device(_DEVICE)
    ref_fn = compiled_lib["ref_fn"]
    torch_dtype = TORCH_DTYPES[compiled_lib["dtype"]]

    kernel = _lib_to_func_binary_1d(compiled_lib["lib"])

    num_cores = 20 * 2
    tile_size = 1024
    tile_counts = [
        1,
        7,
        num_cores - 1,
        num_cores + 3,
        2 * num_cores + 7,
        5 * num_cores - 5,
    ]
    shape_list = [tile_size * tiles for tiles in tile_counts]

    for shape in shape_list:
        x = _make_input(shape, _DEVICE, torch_dtype)
        y = _make_input(shape, _DEVICE, torch_dtype)
        z = torch.empty(shape, device=_DEVICE, dtype=torch_dtype)
        kernel(x, y, z)
        torch.npu.synchronize()
        torch.testing.assert_close(z, ref_fn(x, y))


@pytest.mark.require_npu
def test_binary_2d_precision(compiled_lib):
    torch.npu.set_device(_DEVICE)
    ref_fn = compiled_lib["ref_fn"]
    torch_dtype = TORCH_DTYPES[compiled_lib["dtype"]]

    kernel = _lib_to_func_binary_2d(compiled_lib["lib"])

    num_cores = 20 * 2
    tile_size = 1024
    shape_list = [
        (1, tile_size),
        (3, tile_size),
        (13, 2 * tile_size),
        (100, tile_size),
        (num_cores + 3, 3 * tile_size),
        (2 * num_cores + 7, tile_size),
    ]

    for shape in shape_list:
        x = _make_input(shape, _DEVICE, torch_dtype)
        y = _make_input(shape, _DEVICE, torch_dtype)
        z = torch.empty(shape, device=_DEVICE, dtype=torch_dtype)
        kernel(x, y, z)
        torch.npu.synchronize()
        torch.testing.assert_close(z, ref_fn(x, y))
