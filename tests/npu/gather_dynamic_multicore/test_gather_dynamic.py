import os
import ctypes
import subprocess

import pytest
import torch
from ptodsl.test_util import get_test_device

torch.manual_seed(0)

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEVICE = get_test_device()

TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "int16": torch.int16,
    "int32": torch.int32,
}

# One kernel compiled per (dtype, mask_pattern).
CASES = [
    ("float16", "P0101"),
    ("float16", "P1111"),
    ("float16", "P0001"),
    ("float16", "P1010")
]

# Runtime shapes (B, N). N must be a multiple of 32.
SHAPES = [(1, 64), (8, 64), (4, 128), (2, 256)]

NUM_BLOCKS = 20
TILE = 32


def _case_id(dtype, mask_pattern):
    return f"vec_gather_2d_dynamic_{dtype}_{mask_pattern}"


_PARAMS = [
    pytest.param((dtype, mask_pattern), id=f"{dtype}-{mask_pattern}")
    for dtype, mask_pattern in CASES
]


def _lib_path(dtype, mask_pattern):
    return os.path.join(_DIR, f"{_case_id(dtype, mask_pattern)}_lib.so")


def _ctypes_ptr(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


@pytest.fixture(scope="session", params=_PARAMS)
def compiled_lib(request):
    dtype, mask_pattern = request.param
    subprocess.check_call(
        ["bash", os.path.join(_DIR, "compile.sh"), dtype, mask_pattern],
        cwd=_DIR,
    )
    yield {"dtype": dtype, "mask_pattern": mask_pattern}
    os.remove(_lib_path(dtype, mask_pattern))


def _make_src(shape, device, torch_dtype):
    if torch_dtype.is_floating_point:
        return torch.rand(shape, device=device, dtype=torch_dtype) + 0.1
    return torch.randint(0, 1000, shape, device=device, dtype=torch_dtype)


def init_indices_per_block(B, N, num_blocks, device):
    """
    Create per-tile within-tile element indices for each block's assigned tile range.

    Indices are element indices within each tile: values in [0, TILE-1].
    Per-tile semantics: out[base + j] = src[base + indices[base + j]]
    """
    assert N % TILE == 0
    total = B * N
    num_tiles = total // TILE
    tiles_per_block = (num_tiles + num_blocks - 1) // num_blocks

    flat = torch.empty((total,), device=device, dtype=torch.int32)

    for bid in range(num_blocks):
        start_tile = bid * tiles_per_block
        end_tile = min(num_tiles, start_tile + tiles_per_block)
        if start_tile >= end_tile:
            continue

        start = start_tile * TILE
        end = end_tile * TILE

        flat[start:end] = torch.randint(
            0, TILE, (end - start,), device=device, dtype=torch.int32
        )

    return flat.view(B, N).to(torch.int32)


def _gather_ref_blocked(src, indices, mask_pattern, num_blocks=NUM_BLOCKS):
    """
    Reference matching the kernel partitioning by blocks (NUM_BLOCKS).
    Indices are within-tile element indices in [0, TILE-1].

    Per-tile semantics: out[base + j] = src[base + indices[base + j]]
    """
    B, N = src.shape
    assert N % TILE == 0

    total = B * N
    num_tiles = total // TILE
    tiles_per_block = (num_tiles + num_blocks - 1) // num_blocks

    flat_src = src.reshape(-1)
    flat_idx = indices.reshape(-1).to(torch.int64)

    flat_out = torch.empty_like(flat_src)

    for bid in range(num_blocks):
        start_tile = bid * tiles_per_block
        end_tile = min(num_tiles, start_tile + tiles_per_block)
        if start_tile >= end_tile:
            continue

        tile_ids = torch.arange(
            start_tile, end_tile, device=src.device, dtype=torch.int64
        )
        base = (tile_ids * TILE).repeat_interleave(TILE)
        lane = torch.arange(TILE, device=src.device, dtype=torch.int64).repeat(
            end_tile - start_tile
        )

        pos = base + lane
        src_pos = base + flat_idx[pos]
        gathered = flat_src[src_pos]

        if mask_pattern == "P1111":
            flat_out[pos] = gathered
        elif mask_pattern == "P0101":
            flat_out[pos] = 0
            keep = lane % 2 == 0
            packed_lane = lane // 2
            packed_pos = base + packed_lane
            flat_out[packed_pos[keep]] = gathered[keep]
        elif mask_pattern == "P1010":
            flat_out[pos] = 0
            keep = lane % 2 == 1
            packed_lane = lane // 2
            packed_pos = base + packed_lane
            flat_out[packed_pos[keep]] = gathered[keep]
        elif mask_pattern == "P0001":
            flat_out[pos] = 0
            keep = lane % 4 == 0
            packed_lane = lane // 4
            packed_pos = base + packed_lane
            flat_out[packed_pos[keep]] = gathered[keep]
        else:
            raise ValueError(mask_pattern)

    return flat_out.view(B, N)


def test_build_gather(compiled_lib):
    dtype, mask_pattern = compiled_lib["dtype"], compiled_lib["mask_pattern"]
    assert os.path.exists(_lib_path(dtype, mask_pattern))


@pytest.mark.require_npu
def test_gather_dynamic(compiled_lib):
    import torch_npu

    torch.npu.set_device(_DEVICE)
    dtype = compiled_lib["dtype"]
    mask_pattern = compiled_lib["mask_pattern"]
    torch_dtype = TORCH_DTYPES[dtype]

    lib = ctypes.CDLL(_lib_path(dtype, mask_pattern))
    fn = getattr(lib, f"call_{_case_id(dtype, mask_pattern)}")
    stream_ptr = torch.npu.current_stream()._as_parameter_

    for B, N in SHAPES:
        src = _make_src((B, N), _DEVICE, torch_dtype)

        # indices are within-tile element indices in [0, TILE-1]
        indices = init_indices_per_block(B, N, NUM_BLOCKS, _DEVICE)

        out = torch.empty((B, N), device=_DEVICE, dtype=torch_dtype)

        torch.npu.synchronize()
        fn(stream_ptr, _ctypes_ptr(src), _ctypes_ptr(indices), _ctypes_ptr(out), B, N)
        torch.npu.synchronize()

        ref = _gather_ref_blocked(src, indices, mask_pattern, num_blocks=NUM_BLOCKS)

        torch.testing.assert_close(
            out, ref, msg=f"shape=({B},{N}), mask={mask_pattern}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
