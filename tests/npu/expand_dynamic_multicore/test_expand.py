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
    "colexpand",
    "colexpand_sub",
    "colexpand_div",
    "colexpand_mul",
    "colexpand_min",
    "colexpand_max",
    "rowexpand",
    "rowexpand_add",
    "rowexpand_mul",
    "rowexpand_sub",
    "rowexpand_div",
]

_LIB_PATHS = {name: os.path.join(_DIR, f"{name}_lib.so") for name in _KERNELS}

_SHAPES = [
    (1, 1),
    (7, 7),
    (15, 17),
    (31, 33),
    (33, 31),
    (64, 32),
    (32, 64),
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


_FUSED_KERNELS = {
    "colexpand_sub",
    "colexpand_div",
    "colexpand_mul",
    "colexpand_min",
    "colexpand_max",
    "rowexpand_add",
    "rowexpand_mul",
    "rowexpand_sub",
    "rowexpand_div",
}


def _load_kernel(name):
    lib = ctypes.CDLL(_LIB_PATHS[name])
    fn = getattr(lib, f"call_{name}")
    if name in _FUSED_KERNELS:
        # fused: (blockDim, stream, x, y, z, batch, n_cols)
        fn.argtypes = [
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
    else:
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


def _make_inputs(name, batch, n_cols, device):
    if name == "colexpand":
        src = torch.randn(n_cols, device=device, dtype=torch.float32)
        dst = torch.zeros((batch, n_cols), device=device, dtype=torch.float32)
        return src, dst, None
    if name == "rowexpand":
        src = torch.randn(batch, device=device, dtype=torch.float32)
        dst = torch.zeros((batch, n_cols), device=device, dtype=torch.float32)
        return src, dst, None
    if name == "colexpand_div":
        # avoid division by zero: keep x away from 0
        x = torch.empty(n_cols, device=device, dtype=torch.float32).uniform_(0.5, 1.5)
        y = torch.randn(batch, n_cols, device=device, dtype=torch.float32)
        z = torch.zeros((batch, n_cols), device=device, dtype=torch.float32)
        return x, y, z
    if name in {"colexpand_sub", "colexpand_mul", "colexpand_min", "colexpand_max"}:
        x = torch.randn(n_cols, device=device, dtype=torch.float32)
        y = torch.randn(batch, n_cols, device=device, dtype=torch.float32)
        z = torch.zeros((batch, n_cols), device=device, dtype=torch.float32)
        return x, y, z
    if name == "rowexpand_div":
        # avoid division by zero: keep x away from 0
        x = torch.empty(batch, device=device, dtype=torch.float32).uniform_(0.5, 1.5)
        y = torch.randn(batch, n_cols, device=device, dtype=torch.float32)
        z = torch.zeros((batch, n_cols), device=device, dtype=torch.float32)
        return x, y, z
    # rowexpand_add, rowexpand_mul, rowexpand_sub
    x = torch.randn(batch, device=device, dtype=torch.float32)
    y = torch.randn(batch, n_cols, device=device, dtype=torch.float32)
    z = torch.zeros((batch, n_cols), device=device, dtype=torch.float32)
    return x, y, z


def _reference(name, x, y):
    if name == "colexpand":
        return x.float().unsqueeze(0).expand_as(y)
    if name == "colexpand_sub":
        return y.float() - x.float().unsqueeze(0)
    if name == "colexpand_div":
        return y.float() / x.float().unsqueeze(0)
    if name == "colexpand_mul":
        return y.float() * x.float().unsqueeze(0)
    if name == "colexpand_min":
        return torch.minimum(y.float(), x.float().unsqueeze(0).expand_as(y))
    if name == "colexpand_max":
        return torch.maximum(y.float(), x.float().unsqueeze(0).expand_as(y))
    if name == "rowexpand":
        return x.float().unsqueeze(1).expand_as(y)
    if name == "rowexpand_add":
        return y.float() + x.float().unsqueeze(1)
    if name == "rowexpand_mul":
        return y.float() * x.float().unsqueeze(1)
    if name == "rowexpand_sub":
        return y.float() - x.float().unsqueeze(1)
    if name == "rowexpand_div":
        return y.float() / x.float().unsqueeze(1)
    raise ValueError(f"Unknown kernel: {name}")


def _tolerances(name):
    if name in {"colexpand", "rowexpand"}:
        return {"atol": 0, "rtol": 0}
    if name in {"colexpand_min", "colexpand_max"}:
        return {"atol": 0, "rtol": 0}
    return {"atol": 1e-4, "rtol": 1e-4}


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

    x, y, z = _make_inputs(name, batch, n_cols, _DEVICE)
    dst_ref = _reference(name, x, y)

    stream_ptr = torch.npu.current_stream()._as_parameter_
    if name in _FUSED_KERNELS:
        fn(
            ctypes.c_uint32(_BLOCK_DIM),
            stream_ptr,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(z.data_ptr()),
            ctypes.c_uint32(batch),
            ctypes.c_uint32(n_cols),
        )
        out = z
    else:
        fn(
            ctypes.c_uint32(_BLOCK_DIM),
            stream_ptr,
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_uint32(batch),
            ctypes.c_uint32(n_cols),
        )
        out = y
    torch.npu.synchronize()

    torch.testing.assert_close(out, dst_ref, **_tolerances(name))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
