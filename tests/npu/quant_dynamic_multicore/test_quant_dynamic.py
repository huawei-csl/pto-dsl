import ctypes
import os
import subprocess

import pytest
import torch

from ptodsl.test_util import get_test_device

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = get_test_device()

VARIANTS = ["sym", "asym"]
_PARAMS = [pytest.param(v, id=v) for v in VARIANTS]

# Shapes: (batch, n_cols). n_cols must be a multiple of builder._TILE_COLS (32).
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


def _lib_path(variant):
    return os.path.join(_DIR, f"quant_{variant}_dynamic_lib.so")


def _ctypes_ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


@pytest.fixture(scope="session")
def compiled_libs():
    subprocess.check_call(["bash", os.path.join(_DIR, "compile.sh")], cwd=_DIR)
    yield
    for v in VARIANTS:
        path = _lib_path(v)
        if os.path.exists(path):
            os.remove(path)


@pytest.mark.parametrize("variant", _PARAMS)
def test_build(compiled_libs, variant):
    assert os.path.exists(_lib_path(variant))


def _load_sym_kernel():
    lib = ctypes.CDLL(_lib_path("sym"))
    fn = lib.call_quant_sym_dynamic
    fn.argtypes = [
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # src
        ctypes.c_void_p,  # fp (inv_scale, broadcast per-row)
        ctypes.c_void_p,  # dst
        ctypes.c_int32,  # batch
        ctypes.c_int32,  # n_cols
    ]
    fn.restype = None
    return fn


def _load_asym_kernel():
    lib = ctypes.CDLL(_lib_path("asym"))
    fn = lib.call_quant_asym_dynamic
    fn.argtypes = [
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # src
        ctypes.c_void_p,  # fp (inv_scale, broadcast per-row)
        ctypes.c_void_p,  # offset (zero_point, broadcast per-row)
        ctypes.c_void_p,  # dst
        ctypes.c_int32,  # batch
        ctypes.c_int32,  # n_cols
    ]
    fn.restype = None
    return fn


def _golden_sym(src, inv_scale):
    """Golden reference matching the hardware path.

    Mirrors fp32_to_int8_sym in the reference golden script:
      scaled    = src * inv_scale              (fp32)
      rounded   = round(scaled)               (fp32)
      via_fp16  = rounded.to(fp16)            (exact for integers in [-128,127])
      dst       = clip(via_fp16, -128, 127)   (int8)
    """
    rounded = torch.round(src * inv_scale)
    via_fp16 = rounded.to(torch.float16)
    return via_fp16.clamp(-128, 127).to(torch.int8)


def _golden_asym(src, inv_scale, zero_point):
    """Golden reference matching the hardware path.

    Mirrors fp32_to_int8_asym in the reference golden script:
      scaled    = src * inv_scale + zero_point  (fp32)
      rounded   = round(scaled)                (fp32)
      via_fp16  = rounded.to(fp16)             (exact for integers in [0,255])
      dst       = clip(via_fp16, 0, 255)       (uint8)
    """
    rounded = torch.round(src * inv_scale + zero_point)
    via_fp16 = rounded.to(torch.float16)
    return via_fp16.clamp(0, 255).to(torch.uint8)


def _make_sym_inputs(batch, n_cols, seed):
    """Generate src and a uniform per-tensor inv_scale tile.

    Uses a single inv_scale = 127 / global_max_abs so that fp[i,j] is
    identical for all elements. The hardware applies one scale per tile
    operation (matching the C++ reference's [rows,1] ParaTile), so a
    uniform fp tensor ensures the golden and hardware agree regardless of
    how the scale is read from the tile.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    src = torch.empty(batch, n_cols, dtype=torch.float32).uniform_(
        -2.0, 2.0, generator=g
    )
    max_abs = float(src.abs().max().clamp(min=1e-7))
    inv_scale = torch.full((batch, n_cols), 127.0 / max_abs, dtype=torch.float32)
    return src, inv_scale


def _make_asym_inputs(batch, n_cols, seed):
    """Generate src and uniform per-tensor inv_scale + zero_point tiles.

    Uses global min/max so that all fp[i,j] and offset[i,j] are identical.
    See _make_sym_inputs for the rationale.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    src = torch.empty(batch, n_cols, dtype=torch.float32).uniform_(
        -2.0, 2.0, generator=g
    )
    src_min = float(src.min())
    src_max = float(src.max())
    scale = max((src_max - src_min) / 255.0, 1e-7)
    inv_scale_val = 1.0 / scale
    zero_point_val = float(torch.tensor(-src_min * inv_scale_val).round().clamp(0, 255))
    inv_scale = torch.full((batch, n_cols), inv_scale_val, dtype=torch.float32)
    zero_point = torch.full((batch, n_cols), zero_point_val, dtype=torch.float32)
    return src, inv_scale, zero_point


@pytest.mark.require_npu
@pytest.mark.parametrize("batch, n_cols", _SHAPES, ids=_SHAPE_IDS)
def test_quant_sym(compiled_libs, batch, n_cols):
    import torch_npu  # noqa: F401

    torch.npu.set_device(_DEVICE)
    fn = _load_sym_kernel()
    stream_ptr = torch.npu.current_stream()._as_parameter_

    src, inv_scale = _make_sym_inputs(batch, n_cols, seed=42)
    golden = _golden_sym(src, inv_scale)

    src_dev = src.to(_DEVICE)
    inv_scale_dev = inv_scale.to(_DEVICE)
    dst_dev = torch.zeros(batch, n_cols, device=_DEVICE, dtype=torch.int8)

    torch.npu.synchronize()
    fn(
        stream_ptr,
        _ctypes_ptr(src_dev),
        _ctypes_ptr(inv_scale_dev),
        _ctypes_ptr(dst_dev),
        ctypes.c_int32(batch),
        ctypes.c_int32(n_cols),
    )
    torch.npu.synchronize()

    got = dst_dev.cpu()
    torch.testing.assert_close(
        got.to(torch.int32),
        golden.to(torch.int32),
        atol=0,
        rtol=0,
        msg=f"sym batch={batch} n_cols={n_cols}",
    )


@pytest.mark.require_npu
@pytest.mark.parametrize("batch, n_cols", _SHAPES, ids=_SHAPE_IDS)
def test_quant_asym(compiled_libs, batch, n_cols):
    import torch_npu  # noqa: F401

    torch.npu.set_device(_DEVICE)
    fn = _load_asym_kernel()
    stream_ptr = torch.npu.current_stream()._as_parameter_

    src, inv_scale, zero_point = _make_asym_inputs(batch, n_cols, seed=99)
    golden = _golden_asym(src, inv_scale, zero_point)

    src_dev = src.to(_DEVICE)
    inv_scale_dev = inv_scale.to(_DEVICE)
    zero_point_dev = zero_point.to(_DEVICE)
    dst_dev = torch.zeros(batch, n_cols, device=_DEVICE, dtype=torch.uint8)

    torch.npu.synchronize()
    fn(
        stream_ptr,
        _ctypes_ptr(src_dev),
        _ctypes_ptr(inv_scale_dev),
        _ctypes_ptr(zero_point_dev),
        _ctypes_ptr(dst_dev),
        ctypes.c_int32(batch),
        ctypes.c_int32(n_cols),
    )
    torch.npu.synchronize()

    got = dst_dev.cpu()
    torch.testing.assert_close(
        got.to(torch.int32),
        golden.to(torch.int32),
        atol=0,
        rtol=0,
        msg=f"asym batch={batch} n_cols={n_cols}",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
