"""Shared pytest hooks for frontend tests.

Auto-skip logic for ``require_ptoas_cli`` and ``require_bisheng`` markers.
Actual tool helpers live in ``tooling.py`` to avoid ambiguous ``conftest``
imports (the repo also has a root ``conftest.py``).
"""

import pytest

from tooling import ptoas_available, bisheng_available  # noqa: F811


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "require_ptoas_cli" in item.keywords and not ptoas_available():
            item.add_marker(pytest.mark.skip(reason="ptoas CLI not found on PATH"))
        if "require_bisheng" in item.keywords and not bisheng_available():
            item.add_marker(
                pytest.mark.skip(
                    reason="bisheng not found on PATH or PTO_LIB_PATH/include missing"
                )
            )
