import os
import ctypes
import subprocess

import pytest
import torch
from ptodsl.npu_info import get_num_cube_cores, get_test_device

torch.manual_seed(0)

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = get_test_device()
_BLOCK_DIM = get_num_cube_cores()

UNARY_OPS = [
    # ("rsqrt", lambda x: x.rsqrt()),
    # ("sqrt", lambda x: x.sqrt()),
    # ("exp", lambda x: x.exp()),
    # ("log", lambda x: x.log()),
    # ("relu", lambda x: x.relu()),
    ("abs", lambda x: x.abs()),
    # ("reciprocal", lambda x: x.reciprocal()),
]

DTYPES = ["float32"]

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
}

_SHAPE_LIST = [
    # (10, 20),
    # (20, 10),
    # (1, 128),
    # (7, 1024),
    # (29, 512),
    # (32, 2048),
    # (65, 4096),
    # (200, 8192),
    (20, 10000),
]


_SHAPE_PARAMS = [
    pytest.param(batch, n_cols, id=f"batch{batch}-cols{n_cols}")
    for batch, n_cols in _SHAPE_LIST
]

_PARAMS = [
    pytest.param((op_name, ref_fn, dtype), id=f"{op_name}-{dtype}")
    for op_name, ref_fn in UNARY_OPS
    for dtype in DTYPES
]


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
        "lib_path": _lib_path(op_name, dtype),
    }
    os.remove(_lib_path(op_name, dtype))


def _make_input(shape, device, dtype, op_name):
    """Return a suitable input tensor for the given op.

    rsqrt: inputs in (1.0, 2.0] — keeps outputs near 1.0
           so float16 absolute error stays within 2e-3.
    sqrt/log: inputs in (0.1, 1.1].
    exp: inputs in (-0.5, 0.5] to avoid float16 overflow.
    relu/abs: inputs in (-1.0, 1.0] to exercise both signs.
    """
    if op_name in {"rsqrt", "reciprocal"}:
        return torch.rand(shape, device=device, dtype=dtype) + 1.0
    elif op_name in {"sqrt", "log"}:
        return torch.rand(shape, device=device, dtype=dtype) + 0.1
    elif op_name == "exp":
        return torch.rand(shape, device=device, dtype=dtype) - 0.5
    else:
        return torch.rand(shape, device=device, dtype=dtype) * 2.0 - 1.0


def _lib_to_func_unary(lib):
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.call_kernel.restype = None

    def fn(x, y):
        stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            ctypes.c_uint32(_BLOCK_DIM),
            stream_ptr,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_int32(x.size(0)),
            ctypes.c_int32(x.size(1)),
        )

    return fn


def _lib_path(op_name, dtype):
    return os.path.join(_DIR, f"{op_name}_{dtype}_lib.so")


def test_build_unary_kernels(compiled_lib):
    assert os.path.exists(_lib_path(compiled_lib["op_name"], compiled_lib["dtype"]))


@pytest.mark.require_npu
@pytest.mark.parametrize("batch, n_cols", _SHAPE_PARAMS)
def test_unary_precision(compiled_lib, batch, n_cols):
    import torch_npu  # noqa: F401

    torch.npu.set_device(_DEVICE)
    op_name = compiled_lib["op_name"]
    ref_fn = compiled_lib["ref_fn"]
    torch_dtype = TORCH_DTYPES[compiled_lib["dtype"]]

    lib = ctypes.CDLL(compiled_lib["lib_path"])
    kernel = _lib_to_func_unary(lib)

    x = _make_input((batch, n_cols), _DEVICE, torch_dtype, op_name)
    y = torch.empty(batch, n_cols, device=_DEVICE, dtype=torch_dtype)
    kernel(x, y)
    torch.npu.synchronize()
    y_ref = ref_fn(x)
    torch.npu.synchronize()
    torch.testing.assert_close(y, y_ref, atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
