"""
Forward-only check: PTODSL-generated kernel vs ``sinkhorn_normalize_ref``.

Tests the batched stack-load kernel (``sinkhorn_batch8_builder.py``), the naive
per-matrix kernel (``sinkhorn_k4_builder.py``), and the v2 port
(``sinkhorn_v2_builder.py``).
"""

import pytest
import torch
import torch_npu  # noqa: F401

from jit_util_sinkhorn import sinkhorn_normalize, sinkhorn_normalize_ref


def _generate(n0: int, n1: int, mhc: int, device: str):
    return {
        "comb_res_mix": torch.randn(
            (n0, n1, mhc, mhc), dtype=torch.float16, device=device
        ),
        "repeat": 10,
        "eps": 1e-6,
    }


@pytest.fixture(scope="session")
def device(npu_device):
    return npu_device


@pytest.mark.parametrize("impl", ["batched", "naive", "v2"])
@pytest.mark.parametrize("n0", [1, 2])
@pytest.mark.parametrize("n1", [1, 1024, 4096])
@pytest.mark.parametrize("mhc", [4])
def test_sinkhorn_comprehensive(device, impl, n0, n1, mhc):
    torch.manual_seed(0)
    test_data = _generate(n0=n0, n1=n1, mhc=mhc, device=device)
    x = test_data["comb_res_mix"].clone()

    out_pto = sinkhorn_normalize(
        x, test_data["repeat"], test_data["eps"], impl=impl
    )
    out_ref = sinkhorn_normalize_ref(x, test_data["repeat"], test_data["eps"])

    torch.testing.assert_close(out_pto, out_ref, rtol=1e-2, atol=1e-5)
