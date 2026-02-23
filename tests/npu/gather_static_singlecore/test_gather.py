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
    "int16": torch.int16,
    "int32": torch.int32,
}

# (dtype, (rows, cols), mask_pattern)
# cols must be a multiple of 32; rows*cols must be divisible by 4 (always true here)
CASES = [
    ("float32", (32, 32), "P1111"),
    ("float32", (16, 64), "P1111"),
    ("float32", (16, 128), "P1111"),
    ("float32", (53, 160), "P1111"),
    ("float32", (53, 192), "P1111"),
    ("float16", (32, 32), "P1111"),
    ("float16", (32, 128), "P1111"),
    ("float16", (32, 160), "P1111"),
    ("float16", (3, 160), "P1111"),
    # P0101: positions 0,2 from each group of 4 (every other element) — N//2 valid
    ("float32", (32, 64), "P0101"),
    ("float32", (16, 128), "P0101"),
    ("float16", (32, 64), "P0101"),
    ("float16", (3, 160), "P0101"),
    # P0001: position 0 only from each group of 4 — N//4 valid
    ("float32", (32, 64), "P0001"),
    ("float32", (16, 128), "P0001"),
    ("float16", (32, 64), "P0001"),
    ("float16", (3, 160), "P0001"),
    # int16
    ("int16", (1, 32), "P1111"),
    ("int16", (7, 96), "P1111"),
    ("int16", (53, 160), "P1111"),
    ("int16", (13, 64), "P0101"),
    ("int16", (37, 128), "P0101"),
    ("int16", (200, 64), "P0001"),
    # int32
    ("int32", (77, 128), "P1111"),
    ("int32", (9, 96), "P0101"),
    ("int32", (85, 192), "P0101"),
    ("int32", (3, 256), "P0001"),
    ("int32", (41, 64), "P0001"),
]


def _case_id(dtype, rows, cols, mask_pattern="P1111"):
    return f"{dtype}_{rows}x{cols}_{mask_pattern}"


_PARAMS = [
    pytest.param(
        (dtype, rows, cols, mask_pattern), id=_case_id(dtype, rows, cols, mask_pattern)
    )
    for dtype, (rows, cols), mask_pattern in CASES
]


def _lib_path(dtype, rows, cols, mask_pattern="P1111"):
    return os.path.join(_DIR, f"{_case_id(dtype, rows, cols, mask_pattern)}_lib.so")


def _ctypes_ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


@pytest.fixture(scope="session", params=_PARAMS)
def compiled_lib(request):
    dtype, rows, cols, mask_pattern = request.param
    subprocess.check_call(
        [
            "bash",
            os.path.join(_DIR, "compile.sh"),
            dtype,
            str(rows),
            str(cols),
            mask_pattern,
        ],
        cwd=_DIR,
    )
    yield {"dtype": dtype, "rows": rows, "cols": cols, "mask_pattern": mask_pattern}
    os.remove(_lib_path(dtype, rows, cols, mask_pattern))


def _make_src(shape, device, torch_dtype):
    if torch_dtype.is_floating_point:
        return torch.rand(shape, device=device, dtype=torch_dtype) + 0.1
    return torch.randint(0, 1000, shape, device=device, dtype=torch_dtype)


def _gather_ref(src, indices, mask_pattern="P1111"):
    """Compute the expected gather output for a given mask pattern.

    All non-P1111 patterns operate on the flattened tile in groups of 4.
    Bit positions are read right-to-left (position 0 = rightmost character).
    Selected elements are packed consecutively into the start of the output.

    P1111: all 4/4 selected → full tile, shape [rows, cols]
    P0101: positions 0,2 → every other element,  shape [N//2]
    P0001: position 0    → every 4th element,     shape [N//4]
    """
    intermediate = src.reshape(-1)[indices.reshape(-1)].reshape(indices.shape)
    flat = intermediate.reshape(-1)
    if mask_pattern == "P1111":
        return intermediate
    elif mask_pattern == "P0101":
        return flat[::2]
    elif mask_pattern == "P0001":
        return flat[::4]
    raise ValueError(f"Unknown mask_pattern: {mask_pattern}")


def test_build_gather(compiled_lib):
    dtype, rows, cols, mask_pattern = (
        compiled_lib["dtype"],
        compiled_lib["rows"],
        compiled_lib["cols"],
        compiled_lib["mask_pattern"],
    )
    assert os.path.exists(_lib_path(dtype, rows, cols, mask_pattern))


@pytest.mark.require_npu
def test_gather_precision(compiled_lib):
    import torch_npu

    torch.npu.set_device(_DEVICE)
    dtype = compiled_lib["dtype"]
    rows = compiled_lib["rows"]
    cols = compiled_lib["cols"]
    mask_pattern = compiled_lib["mask_pattern"]
    lib = ctypes.CDLL(_lib_path(dtype, rows, cols, mask_pattern))
    torch_dtype = TORCH_DTYPES[dtype]

    fn = getattr(lib, f"call_{_case_id(dtype, rows, cols, mask_pattern)}")
    stream_ptr = torch.npu.current_stream()._as_parameter_

    shape = (rows, cols)
    src = _make_src(shape, _DEVICE, torch_dtype)
    indices = torch.randint(0, rows * cols, shape, device=_DEVICE, dtype=torch.int32)
    out = torch.empty(shape, device=_DEVICE, dtype=torch_dtype)

    torch.npu.synchronize()  # flush any pending work before using the kernel
    fn(stream_ptr, _ctypes_ptr(src), _ctypes_ptr(indices), _ctypes_ptr(out))
    torch.npu.synchronize()

    ref = _gather_ref(src, indices, mask_pattern)
    if mask_pattern == "P1111":
        torch.testing.assert_close(out, ref)
    else:
        # Mask gathers pack selected elements into the first ref.numel() positions
        # of the flattened output; the rest is undefined.
        torch.testing.assert_close(out.reshape(-1)[: ref.numel()], ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
