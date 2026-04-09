import ctypes
import os
import subprocess

import pytest
import torch

from ptodsl.test_util import get_test_device

torch.manual_seed(0)

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = get_test_device()
_BLOCK_DIM = 24

_KERNELS = [
    "rowsum",
    "rowmin",
    "rowmax",
    "rowprod",
    "colsum",
    "colmin",
    "colmax",
    "colprod",
]

_LIB_PATHS = {name: os.path.join(_DIR, f"{name}_lib.so") for name in _KERNELS}

_SHAPES = [
    (1, 1),
    (2, 3),
    (3, 2),
    (7, 7),
    (15, 17),
    (17, 15),
    (31, 32),
    (32, 31),
    (32, 32),
    (33, 33),
    (31, 33),
    (33, 31),
    (64, 32),
    (32, 64),
    (63, 64),
    (64, 63),
    (65, 33),
    (65, 64),
    (29, 257),
    (127, 129),
]

_SHAPE_IDS = [f"batch{batch}-cols{n_cols}" for batch, n_cols in _SHAPES]


@pytest.fixture(scope="session")
def compiled_kernels():
    subprocess.check_call(["bash", os.path.join(_DIR, "compile.sh")], cwd=_DIR)
    yield
    for path in _LIB_PATHS.values():
        if os.path.exists(path):
            os.remove(path)


def _load_kernel(name):
    lib = ctypes.CDLL(_LIB_PATHS[name])
    fn = getattr(lib, f"call_{name}")
    fn.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    fn.restype = None
    return fn


def _reference(name, x):
    if name == "rowsum":
        return x.float().sum(dim=-1)
    if name == "rowmin":
        return x.float().amin(dim=-1)
    if name == "rowmax":
        return x.float().amax(dim=-1)
    if name == "rowprod":
        return x.float().prod(dim=-1)
    if name == "colsum":
        return x.float().sum(dim=0)
    if name == "colmin":
        return x.float().amin(dim=0)
    if name == "colmax":
        return x.float().amax(dim=0)
    if name == "colprod":
        return x.float().prod(dim=0)
    raise ValueError(f"Unknown kernel: {name}")


def _output_shape(name, batch, n_cols):
    return (batch,) if name.startswith("row") else (n_cols,)


def _make_input(name, batch, n_cols, device):
    if name.endswith("prod"):
        return torch.empty(batch, n_cols, device=device, dtype=torch.float32).uniform_(
            0.5, 1.5
        )
    return torch.randn(batch, n_cols, device=device, dtype=torch.float32)


def _tolerances(name):
    if name.endswith("prod"):
        return {"atol": 1e-3, "rtol": 1e-3}
    return {"atol": 1e-4, "rtol": 0}


@pytest.mark.parametrize("name", _KERNELS)
def test_build_kernel(compiled_kernels, name):
    assert os.path.exists(_LIB_PATHS[name])


@pytest.mark.require_npu
@pytest.mark.parametrize("name", _KERNELS)
@pytest.mark.parametrize("batch, n_cols", _SHAPES, ids=_SHAPE_IDS)
def test_kernel_precision(compiled_kernels, name, batch, n_cols):
    import torch_npu  # noqa: F401

    torch.npu.set_device(_DEVICE)

    fn = _load_kernel(name)

    x = _make_input(name, batch, n_cols, _DEVICE)
    y = torch.full(
        _output_shape(name, batch, n_cols),
        float("nan"),
        device=_DEVICE,
        dtype=torch.float32,
    )
    y_ref = _reference(name, x)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    fn(
        ctypes.c_uint32(_BLOCK_DIM),
        stream_ptr,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_uint32(batch),
        ctypes.c_uint32(n_cols),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(y, y_ref, **_tolerances(name))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
