import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "require_npu: marks tests as requiring NPU hardware (deselect with '-m \"not require_npu\"')",
    )
