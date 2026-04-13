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

# TSORT32 sorts within fixed 32-element blocks.
# Each input element expands into (score, index) pairs in the output:
#   float16: dst_stride=4  →  [score_f16, zero, idx_lo_u16, idx_hi_u16]
#   float32: dst_stride=2  →  [score_f32, idx_u32]
# tile_length must be a multiple of SORT_BLOCK_LEN.
TILE_LENGTH = 1024
SORT_BLOCK_LEN = 32
DTYPES = ["float16", "float32"]
SIZES = [1024, 2048, 3072, 4096, 6144, 8192, 16384]

_DST_STRIDE = {"float16": 4, "float32": 2}
_TORCH_DTYPES = {"float16": torch.float16, "float32": torch.float32}

_DTYPE_PARAMS = [pytest.param(dtype, id=dtype) for dtype in DTYPES]
_SIZE_PARAMS = [pytest.param(N, id=f"N{N}") for N in SIZES]


def _fn_name(dtype):
    return f"tsort32_1d_dynamic_{dtype}"


def _lib_path(dtype):
    return os.path.join(_DIR, f"{_fn_name(dtype)}_lib.so")


def _ctypes_ptr(tensor: torch.Tensor):
    return ctypes.c_void_p(tensor.data_ptr())


@pytest.fixture(scope="session", params=_DTYPE_PARAMS)
def compiled_lib(request):
    dtype = request.param
    subprocess.check_call(
        ["bash", os.path.join(_DIR, "compile.sh"), dtype],
        cwd=_DIR,
    )
    yield {"dtype": dtype}
    libp = _lib_path(dtype)
    if os.path.exists(libp):
        os.remove(libp)


def _load_fn(dtype):
    lib = ctypes.CDLL(_lib_path(dtype))
    fn = getattr(lib, f"call_{_fn_name(dtype)}")
    fn.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # src
        ctypes.c_void_p,  # idx  (uint32)
        ctypes.c_void_p,  # dst  (N * dst_stride elements)
        ctypes.c_int32,  # N
    ]
    fn.restype = None
    return fn


def _run_kernel(
    fn, stream_ptr, src: torch.Tensor, idx: torch.Tensor, N: int, dst_stride: int
) -> torch.Tensor:
    import torch_npu

    dst = torch.empty(N * dst_stride, dtype=src.dtype, device=src.device)
    torch.npu.synchronize()
    fn(
        ctypes.c_uint32(_BLOCK_DIM),
        stream_ptr,
        _ctypes_ptr(src),
        _ctypes_ptr(idx),
        _ctypes_ptr(dst),
        ctypes.c_int32(N),
    )
    torch.npu.synchronize()
    return dst


def _check_preconditions(N: int):
    assert (
        N % TILE_LENGTH == 0
    ), f"N must be a multiple of TILE_LENGTH={TILE_LENGTH}, got {N}"
    assert TILE_LENGTH % SORT_BLOCK_LEN == 0


def _extract_scores(dst: torch.Tensor, dst_stride: int) -> torch.Tensor:
    """Slot 0 of each output group holds the sorted score."""
    return dst.cpu().reshape(-1, dst_stride)[:, 0]


def _reference_scores(src: torch.Tensor) -> torch.Tensor:
    """Sort each SORT_BLOCK_LEN-element group descending."""
    return (
        src.cpu()
        .reshape(-1, SORT_BLOCK_LEN)
        .sort(dim=1, descending=True)
        .values.reshape(-1)
    )


def test_build_tsort32(compiled_lib):
    dtype = compiled_lib["dtype"]
    assert os.path.exists(_lib_path(dtype))


@pytest.mark.require_npu
@pytest.mark.parametrize("N", _SIZE_PARAMS)
def test_tsort32_scores(compiled_lib, N):
    """Scores extracted from TSORT32 output match per-block sorted input."""
    import torch_npu

    dtype = compiled_lib["dtype"]
    torch_dtype = _TORCH_DTYPES[dtype]
    dst_stride = _DST_STRIDE[dtype]

    _check_preconditions(N)
    torch.npu.set_device(_DEVICE)

    fn = _load_fn(dtype)
    stream_ptr = torch.npu.current_stream()._as_parameter_

    src = torch.rand(N, dtype=torch_dtype, device=_DEVICE)
    idx = torch.arange(N, dtype=torch.int32, device=_DEVICE)
    dst = _run_kernel(fn, stream_ptr, src, idx, N, dst_stride)

    scores_got = _extract_scores(dst, dst_stride)
    scores_ref = _reference_scores(src)

    torch.testing.assert_close(
        scores_got,
        scores_ref,
        msg="TSORT32 scores do not match per-block sorted reference",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
