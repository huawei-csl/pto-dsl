import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "require_npu: marks tests as requiring NPU hardware (deselect with '-m \"not require_npu\"')",
    )
    config.addinivalue_line(
        "markers",
        "require_ptoas_cli: marks tests as requiring the ptoas CLI on PATH",
    )
    config.addinivalue_line(
        "markers",
        "require_bisheng: marks tests as requiring the bisheng compiler and PTO_LIB_PATH",
    )
