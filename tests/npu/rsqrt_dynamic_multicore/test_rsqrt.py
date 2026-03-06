import ctypes
import os
import subprocess

import pytest
import torch

from ptodsl.test_util import get_test_device

torch.manual_seed(0)

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = get_test_device()

_DTYPES = ["float32", "float16"]
_TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16}

_BATCH_LIST = [1, 7, 29, 32, 65, 200]
_N_COLS_LIST = [128, 256, 512, 1024, 2048, 4096, 8192]

_SHAPE_PARAMS = [
    pytest.param(dtype, batch, n_cols, id=f"{dtype}-batch{batch}-cols{n_cols}")
    for dtype in _DTYPES
    for batch in _BATCH_LIST
    for n_cols in _N_COLS_LIST
]


def _lib_path(dtype):
    return os.path.join(_DIR, f"rsqrt_{dtype}_lib.so")


@pytest.fixture(scope="session")
def compiled_rsqrt():
    for dtype in _DTYPES:
        subprocess.check_call(
            ["bash", os.path.join(_DIR, "compile.sh"), dtype],
            cwd=_DIR,
        )
    yield
    for dtype in _DTYPES:
        os.remove(_lib_path(dtype))


def test_build_rsqrt(compiled_rsqrt):
    for dtype in _DTYPES:
        assert os.path.exists(_lib_path(dtype))


@pytest.mark.require_npu
@pytest.mark.parametrize("dtype, batch, n_cols", _SHAPE_PARAMS)
def test_rsqrt_precision(compiled_rsqrt, dtype, batch, n_cols):
    import torch_npu  # noqa: F401

    torch_dtype = _TORCH_DTYPES[dtype]
    lib = ctypes.CDLL(_lib_path(dtype))
    lib.call_kernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.call_kernel.restype = None

    torch.npu.set_device(_DEVICE)
    x = torch.rand(batch, n_cols, device=_DEVICE, dtype=torch_dtype) + 1.0
    y = torch.empty(batch, n_cols, device=_DEVICE, dtype=torch_dtype)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        stream_ptr,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_int32(batch),
        ctypes.c_int32(n_cols),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(y, x.rsqrt(), atol=2e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
