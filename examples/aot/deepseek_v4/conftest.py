"""Shared pytest config for deepseek_v4 example tests.

Provides an ``npu_device`` fixture (mirrors ``examples/aot/sinkhorn_demo``).
Each per-op test directory adds its own folder to ``sys.path`` via its own
``conftest.py`` so its ``*_util.py`` is importable when pytest is launched
from the repo root.
"""

import sys
from pathlib import Path

import pytest
import torch


def normalize_npu_device(device) -> str:
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
            help="NPU device (examples: 0, npu:0).",
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


# Make per-op directories importable when invoking ``pytest`` from repo root.
_HERE = Path(__file__).resolve().parent
for sub in _HERE.iterdir():
    if sub.is_dir() and (sub / "__init__.py").exists() is False:
        if str(sub) not in sys.path:
            sys.path.insert(0, str(sub))
