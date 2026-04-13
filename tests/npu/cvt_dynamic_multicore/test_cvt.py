import ctypes
import os
import subprocess

import pytest
import torch

from ptodsl.npu_info import get_num_cube_cores, get_test_device

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = get_test_device()
_BLOCK_DIM = get_num_cube_cores()

# ---------------------------------------------------------------------------
# Dtype metadata
# ---------------------------------------------------------------------------

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

_FLOAT_DTYPES = {"float32", "float16"}
_INT_DTYPES = {"int64", "int32", "int16", "int8", "uint8"}

# ---------------------------------------------------------------------------
# Conversion matrix
# ---------------------------------------------------------------------------

# (src_dtype, dst_dtype, rmode)  — None rmode = no rounding-mode attribute
# Three conversions covering float narrowing, float→int (with int8), and int→int.
_CONVERSIONS = [
    # float narrowing — floor (toward -∞)
    ("float32", "float16", "floor"),
    # float16 → small signed int — cast_rint
    ("float16", "int8", "cast_rint"),
    # int → int narrowing (no rmode)
    ("int32", "int16", None),
]

_CONV_IDS = [f"{s}_to_{d}_{r}" if r else f"{s}_to_{d}" for s, d, r in _CONVERSIONS]

# Shapes: (batch, n_cols).  n_cols must be a multiple of builder._TILE_COLS (32).
_SHAPES = [
    (4, 32),
    (7, 64),
    (32, 32),
    (32, 128),
    (128, 128),
    (15, 96),
    (33, 128),
]
_SHAPE_IDS = [f"batch{b}-cols{c}" for b, c in _SHAPES]

# ---------------------------------------------------------------------------
# Helpers: lib paths, loaders, golden computation
# ---------------------------------------------------------------------------

_EPS_F16 = torch.finfo(torch.float16).eps
_TINY_F16 = torch.finfo(torch.float16).smallest_normal


def _lib_path(src_dtype, dst_dtype, rmode=None):
    stem = f"cvt_{src_dtype}_to_{dst_dtype}" + (f"_{rmode}" if rmode else "")
    return os.path.join(_DIR, f"{stem}_lib.so")


def _load_kernel(src_dtype, dst_dtype, rmode=None):
    lib = ctypes.CDLL(_lib_path(src_dtype, dst_dtype, rmode))
    fn = getattr(lib, f"call_cvt_{src_dtype}_to_{dst_dtype}")
    fn.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # src
        ctypes.c_void_p,  # dst
        ctypes.c_int32,  # batch
        ctypes.c_int32,  # n_cols
    ]
    fn.restype = None
    return fn


def _make_input(src_dtype, batch, n_cols, seed, dst_dtype=None):
    """Generate a small-range input that avoids overflow for all dst types."""
    g = torch.Generator()
    g.manual_seed(seed)
    dtype = _TORCH_DTYPE[src_dtype]
    if src_dtype in _FLOAT_DTYPES:
        # uint8 cannot represent negative values, so keep inputs non-negative.
        # For all other integer destinations [-100, 100] fits in int8 ([-128,127]).
        if dst_dtype == "uint8":
            return (
                torch.empty(batch, n_cols).uniform_(0.0, 100.0, generator=g).to(dtype)
            )
        # [-100, 100]: fits in int8 ([-128,127]) after rounding
        return torch.empty(batch, n_cols).uniform_(-100.0, 100.0, generator=g).to(dtype)
    elif src_dtype == "uint8":
        return torch.randint(0, 101, (batch, n_cols), generator=g, dtype=dtype)
    else:
        return torch.randint(-100, 101, (batch, n_cols), generator=g, dtype=dtype)


def _compute_golden(src, dst_dtype, rmode):
    """Return (golden_cpu, is_exact).

    is_exact=False signals a 1-ULP bound check (used for f32→f16).
    """
    dst_torch = _TORCH_DTYPE[dst_dtype]

    if dst_dtype in _INT_DTYPES:
        info = torch.iinfo(dst_torch)
        src_f = src.float()
        if rmode == "cast_rint":
            rounded = torch.round(src_f)
        else:
            # "trunc" or None (int→int passes through unmodified)
            rounded = torch.trunc(src_f) if src.dtype.is_floating_point else src_f
        return rounded.clamp(info.min, info.max).to(dst_torch), True

    # float destination
    if dst_dtype == "float16" and src.dtype == torch.float32:
        # Narrowing: result is within 1 ULP, exact golden not computable per mode
        return None, False

    return src.to(dst_torch), True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def compiled_libs():
    subprocess.check_call(["bash", os.path.join(_DIR, "compile.sh")], cwd=_DIR)
    yield
    for src, dst, rmode in _CONVERSIONS:
        path = _lib_path(src, dst, rmode)
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("src_dtype, dst_dtype, rmode", _CONVERSIONS, ids=_CONV_IDS)
def test_build(compiled_libs, src_dtype, dst_dtype, rmode):
    assert os.path.exists(_lib_path(src_dtype, dst_dtype, rmode))


# ---------------------------------------------------------------------------
# Precision tests
# ---------------------------------------------------------------------------


@pytest.mark.require_npu
@pytest.mark.parametrize("src_dtype, dst_dtype, rmode", _CONVERSIONS, ids=_CONV_IDS)
@pytest.mark.parametrize("batch, n_cols", _SHAPES, ids=_SHAPE_IDS)
def test_cvt(compiled_libs, src_dtype, dst_dtype, rmode, batch, n_cols):
    import torch_npu  # noqa: F401

    torch.npu.set_device(_DEVICE)
    fn = _load_kernel(src_dtype, dst_dtype, rmode)
    stream_ptr = torch.npu.current_stream()._as_parameter_

    for i in range(4):
        src = _make_input(src_dtype, batch, n_cols, seed=42 + i, dst_dtype=dst_dtype)
        golden, is_exact = _compute_golden(src, dst_dtype, rmode)

        src_dev = src.to(_DEVICE)
        dst_dev = torch.zeros(
            batch, n_cols, device=_DEVICE, dtype=_TORCH_DTYPE[dst_dtype]
        )

        torch.npu.synchronize()
        fn(
            ctypes.c_uint32(_BLOCK_DIM),
            stream_ptr,
            ctypes.c_void_p(src_dev.data_ptr()),
            ctypes.c_void_p(dst_dev.data_ptr()),
            ctypes.c_int32(batch),
            ctypes.c_int32(n_cols),
        )
        torch.npu.synchronize()

        got = dst_dev.cpu()
        tag = f"{src_dtype}→{dst_dtype} rmode={rmode} trial={i}"

        if is_exact:
            mismatches = int((got != golden).sum())
            assert mismatches == 0, f"{tag}: {mismatches} mismatches"
        else:
            # 1-ULP bound for narrowing float conversions
            ulp = src.abs().clamp(min=_TINY_F16) * _EPS_F16
            bad = int((got.float() - src).abs().gt(ulp).sum())
            assert bad == 0, f"{tag}: {bad} elements exceed 1 ULP"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
