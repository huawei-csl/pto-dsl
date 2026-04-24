"""Local pytest hooks: NPU device selection for this directory only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_baseline_dir = Path(__file__).resolve().parent
if str(_baseline_dir) not in sys.path:
    sys.path.insert(0, str(_baseline_dir))


def normalize_npu_device(device: str | int) -> str:
    text = str(device).strip().strip('"').strip("'")
    if text.lower().startswith("npu:"):
        index = text.split(":", 1)[1].strip()
    else:
        index = text

    if not index.isdigit():
        raise ValueError(
            f"Invalid NPU device '{device}'. Expected values like 0 or npu:0."
        )
    return f"npu:{int(index)}"


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--npu",
            action="store",
            default="npu:0",
            help="NPU device (examples: 0, npu:0, '0', 'npu:0').",
        )
    except ValueError as exc:
        if "--npu" not in str(exc):
            raise


@pytest.fixture(scope="session")
def npu_device(request):
    raw = request.config.getoption("--npu")
    return normalize_npu_device(raw)


@pytest.fixture(scope="session", autouse=True)
def setup_npu_device(npu_device):
    torch.npu.set_device(npu_device)
