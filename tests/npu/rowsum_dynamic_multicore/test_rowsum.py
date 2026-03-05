import ctypes
import os
import subprocess

import pytest
import torch

from ptodsl.test_util import get_test_device

torch.manual_seed(0)

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = get_test_device()
_LIB_PATH = os.path.join(_DIR, "rowsum_lib.so")
_BLOCK_DIM = 24

_BATCH_LIST = [1, 4, 22, 65]
_N_COLS_LIST = [128, 256, 512, 1024, 2048, 4096, 8192]

_SHAPE_PARAMS = [
    pytest.param(batch, n_cols, id=f"batch{batch}-cols{n_cols}")
    for batch in _BATCH_LIST
    for n_cols in _N_COLS_LIST
]


@pytest.fixture(scope="session")
def compiled_rowsum():
    subprocess.check_call(["bash", os.path.join(_DIR, "compile.sh")], cwd=_DIR)
    yield
    os.remove(_LIB_PATH)


def test_build_rowsum(compiled_rowsum):
    assert os.path.exists(_LIB_PATH)


@pytest.mark.require_npu
@pytest.mark.parametrize("batch, n_cols", _SHAPE_PARAMS)
def test_rowsum_precision(compiled_rowsum, batch, n_cols):
    import torch_npu  # noqa: F401

    lib = ctypes.CDLL(_LIB_PATH)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.call_kernel.restype = None

    torch.npu.set_device(_DEVICE)
    x = torch.randn(batch, n_cols, device=_DEVICE, dtype=torch.float32)
    y = torch.empty(batch, device=_DEVICE, dtype=torch.float32)

    y_ref = x.float().sum(dim=-1)
    stream_ptr = torch.npu.current_stream()._as_parameter_
    lib.call_kernel(
        ctypes.c_uint32(_BLOCK_DIM),
        stream_ptr,
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(y.data_ptr()),
        ctypes.c_uint32(batch),
        ctypes.c_uint32(n_cols),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
