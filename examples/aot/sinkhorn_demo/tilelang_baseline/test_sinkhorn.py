"""TileLang Sinkhorn kernel vs ``sinkhorn_normalize_ref`` (same directory)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))

import tilelang  # noqa: E402
import torch_npu  # noqa: F401, E402

from sinkhorn_kernel import build_sinkhorn, sinkhorn_normalize_ref  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _tilelang_no_cache():
    tilelang.disable_cache()
    yield


def _generate(n0: int, n1: int, mhc: int, device: str):
    return {
        "x": torch.randn((n0, n1, mhc, mhc), dtype=torch.float32, device=device),
        "repeat": 10,
        "eps": 1e-6,
    }


@pytest.mark.parametrize("n0", [1, 2])
@pytest.mark.parametrize("n1", [1, 1024, 4096])
@pytest.mark.parametrize("mhc", [4])
def test_sinkhorn_tilelang_vs_ref(npu_device, n0, n1, mhc):
    torch.manual_seed(0)
    td = _generate(n0=n0, n1=n1, mhc=mhc, device=npu_device)
    x = td["x"].clone()
    nmat = n0 * n1
    x_flat = x.reshape(nmat, mhc, mhc).contiguous()

    fn = build_sinkhorn(hc=mhc, sinkhorn_iters=td["repeat"], eps=td["eps"])
    out_flat = fn(x_flat)
    torch.npu.synchronize()

    ref = sinkhorn_normalize_ref(
        x_flat.cpu(), td["repeat"], td["eps"]
    ).to(device=npu_device, dtype=torch.float32)

    torch.testing.assert_close(out_flat, ref, rtol=1e-2, atol=1e-2)
