import os
import ctypes
import subprocess

import pytest
import torch

torch.manual_seed(0)

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = "npu:6"

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
}

# (dtype, (rows, cols)) — edit here to add/remove cases, first dim can be any, the second a multiple of 32
CASES = [
    ("float32", (32, 32)),
    ("float32", (16, 64)),
    ("float32", (16, 128)),
    ("float32", (53, 160)),
    ("float32", (53, 192)),
    ("float16", (32, 32)),
    ("float16", (32, 128)),
    ("float16", (32, 160)),
    ("float16", (3, 160)),
]


def _case_id(dtype, rows, cols):
    return f"{dtype}_{rows}x{cols}"


_PARAMS = [
    pytest.param((dtype, rows, cols), id=_case_id(dtype, rows, cols))
    for dtype, (rows, cols) in CASES
]


def _lib_path(dtype, rows, cols):
    return os.path.join(_DIR, f"{_case_id(dtype, rows, cols)}_lib.so")


def _ctypes_ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


@pytest.fixture(scope="session", params=_PARAMS)
def compiled_lib(request):
    dtype, rows, cols = request.param
    subprocess.check_call(
        ["bash", os.path.join(_DIR, "compile.sh"), dtype, str(rows), str(cols)],
        cwd=_DIR,
    )
    yield {"dtype": dtype, "rows": rows, "cols": cols}
    os.remove(_lib_path(dtype, rows, cols))


def _make_src(shape, device, torch_dtype):
    if torch_dtype.is_floating_point:
        return torch.rand(shape, device=device, dtype=torch_dtype) + 0.1
    return torch.randint(0, 1000, shape, device=device, dtype=torch_dtype)


def _gather_ref(src, indices):
    """out[i,j] = src.flatten()[indices[i,j]]"""
    return src.reshape(-1)[indices.reshape(-1)].reshape(indices.shape)


def test_build_gather(compiled_lib):
    dtype, rows, cols = (
        compiled_lib["dtype"],
        compiled_lib["rows"],
        compiled_lib["cols"],
    )
    assert os.path.exists(_lib_path(dtype, rows, cols))


@pytest.mark.require_npu
def test_gather_precision(compiled_lib):
    torch.npu.set_device(_DEVICE)
    dtype = compiled_lib["dtype"]
    rows = compiled_lib["rows"]
    cols = compiled_lib["cols"]
    lib = ctypes.CDLL(_lib_path(dtype, rows, cols))
    torch_dtype = TORCH_DTYPES[dtype]

    fn = getattr(lib, f"call_{_case_id(dtype, rows, cols)}")
    stream_ptr = torch.npu.current_stream()._as_parameter_

    shape = (rows, cols)
    src = _make_src(shape, _DEVICE, torch_dtype)
    indices = torch.randint(0, rows * cols, shape, device=_DEVICE, dtype=torch.int32)
    out = torch.empty(shape, device=_DEVICE, dtype=torch_dtype)

    torch.npu.synchronize()  # flush any pending work before using the kernel
    fn(stream_ptr, _ctypes_ptr(src), _ctypes_ptr(indices), _ctypes_ptr(out))
    torch.npu.synchronize()

    ref = _gather_ref(src, indices)
    torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
