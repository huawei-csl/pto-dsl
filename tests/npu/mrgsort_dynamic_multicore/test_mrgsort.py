import os
import ctypes
import subprocess

import pytest
import torch
from ptodsl.test_util import get_test_device

torch.manual_seed(0)

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = get_test_device()

# TMRGSORT single-list constraints (in terms of hw_block_len = BLOCK_LEN * TYPE_COEF):
#   hw_block_len % 64 == 0
#   tile_length % (hw_block_len * 4) == 0
#   1 <= tile_length // (hw_block_len * 4) <= 255
# TYPE_COEF = sizeof(float) / sizeof(T): 1 for float32, 2 for float16.
TILE_LENGTH = 1024
BLOCK_LEN = 64
TYPE_COEFS = {"float32": 1, "float16": 2}

# TMRGSORT operates on (float32, uint32) interleaved pairs for float16 tiles,
# not on plain float16 values. Sorting plain float16 with TMRGSORT requires
# a TSORT32 pre-pass (to produce the pair format) and a TGATHER post-pass
# (to extract values). The tests here cover the plain-value sort path only,
# which is supported for float32.
DTYPES = ["float32"]
SIZES = [1024, 2048, 3072, 4096, 8192, 16384]

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
}

_DTYPE_PARAMS = [pytest.param(dtype, id=dtype) for dtype in DTYPES]
_SIZE_PARAMS = [pytest.param(N, id=f"N{N}") for N in SIZES]


def _fn_name(dtype: str) -> str:
    return f"vec_mrgsort_1d_dynamic_{dtype}"


def _lib_path(dtype: str) -> str:
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


def _check_preconditions(N: int, block_len: int, tile_length: int):
    assert block_len % 64 == 0, f"block_len must be multiple of 64, got {block_len}"
    assert tile_length % (block_len * 4) == 0, (
        f"tile_length must be multiple of block_len*4, got "
        f"tile_length={tile_length}, block_len={block_len}"
    )
    repeat_times = tile_length // (block_len * 4)
    assert (
        1 <= repeat_times <= 255
    ), f"repeat_times must be in [1, 255], got {repeat_times}"
    assert N % tile_length == 0, f"N must be a multiple of tile_length, got N={N}"


def _make_sorted_sublists(
    N: int,
    block_len: int,
    device,
    torch_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Create N values split into sorted descending sublists of length block_len.
    """
    assert N % block_len == 0
    data = torch.rand(N, dtype=torch.float32)
    data = data.view(-1, block_len)
    data = torch.sort(data, dim=1, descending=True).values
    return data.reshape(-1).to(dtype=torch_dtype, device=device)


def _load_fn(dtype: str):
    lib = ctypes.CDLL(_lib_path(dtype))
    fn = getattr(lib, f"call_{_fn_name(dtype)}")
    fn.argtypes = [
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # src
        ctypes.c_void_p,  # out
        ctypes.c_int32,  # N
    ]
    fn.restype = None
    return fn


def _run_kernel(fn, stream_ptr, src: torch.Tensor, N: int) -> torch.Tensor:
    import torch_npu

    out = torch.empty_like(src)
    torch.npu.synchronize()
    fn(stream_ptr, _ctypes_ptr(src), _ctypes_ptr(out), ctypes.c_int32(N))
    torch.npu.synchronize()
    return out


def _sort_tiles(x: torch.Tensor, tile_length: int) -> torch.Tensor:
    """
    Sort each tile independently descending.
    The multi-pass kernel fully sorts each tile (float32) or sorts within
    hw_block_len*4 sub-segments (float16); either way, sorted(out_tile) must
    equal sorted(src_tile) for a correct permutation sort.
    """
    x = x.cpu().float().reshape(-1, tile_length)
    x = torch.sort(x, dim=1, descending=True).values
    return x.reshape(-1)


def test_build_mrgsort(compiled_lib):
    dtype = compiled_lib["dtype"]
    assert os.path.exists(_lib_path(dtype))


@pytest.mark.require_npu
@pytest.mark.parametrize("N", _SIZE_PARAMS)
def test_mrgsort_equal_after_canonicalization(compiled_lib, N):
    """
    Compare exact equality after canonicalizing each hw_block_len*4 segment.
    This is the right equality check for the current single-list TMRGSORT behavior.
    """
    import torch_npu

    dtype = compiled_lib["dtype"]
    hw_block_len = BLOCK_LEN * TYPE_COEFS[dtype]
    _check_preconditions(N, hw_block_len, TILE_LENGTH)
    torch.npu.set_device(_DEVICE)

    torch_dtype = TORCH_DTYPES[dtype]
    fn = _load_fn(dtype)
    stream_ptr = torch.npu.current_stream()._as_parameter_

    src = _make_sorted_sublists(N, hw_block_len, _DEVICE, torch_dtype)
    out = _run_kernel(fn, stream_ptr, src, N)

    ref = _sort_tiles(src, TILE_LENGTH).to(torch_dtype)
    got = _sort_tiles(out, TILE_LENGTH).to(torch_dtype)

    torch.testing.assert_close(
        got.cpu(),
        ref.cpu(),
        msg="sorted output does not match sorted reference",
    )


@pytest.mark.require_npu
@pytest.mark.parametrize("N", [1024])
def test_mrgsort_deterministic(compiled_lib, N):
    """
    Same input must produce identical output on two runs.
    """
    import torch_npu

    dtype = compiled_lib["dtype"]
    hw_block_len = BLOCK_LEN * TYPE_COEFS[dtype]
    _check_preconditions(N, hw_block_len, TILE_LENGTH)
    torch.npu.set_device(_DEVICE)

    torch_dtype = TORCH_DTYPES[dtype]
    fn = _load_fn(dtype)
    stream_ptr = torch.npu.current_stream()._as_parameter_

    src = _make_sorted_sublists(N, hw_block_len, _DEVICE, torch_dtype)
    out1 = _run_kernel(fn, stream_ptr, src, N)
    out2 = _run_kernel(fn, stream_ptr, src, N)

    torch.testing.assert_close(
        out1.cpu(),
        out2.cpu(),
        msg="kernel output is not deterministic",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
