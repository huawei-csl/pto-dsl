import difflib
import importlib

import ptodsl
import pytest

from ._contract import (
    MIRROR_MODULES,
    PUBLIC_MODULES,
    collect_public_api_snapshot,
    exported_names,
    load_snapshot,
    snapshot_json,
)


def test_committed_public_api_snapshot_matches_current_exports():
    expected = load_snapshot()
    actual = collect_public_api_snapshot()

    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                snapshot_json(expected).splitlines(keepends=True),
                snapshot_json(actual).splitlines(keepends=True),
                fromfile="tests/api/public_api_snapshot.json",
                tofile="current-public-api",
            )
        )
        pytest.fail(
            "Public API snapshot drifted.\n"
            "Run `python scripts/update_public_api_snapshot.py` and commit the result.\n\n"
            f"{diff}"
        )


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_all_entries_in___all___resolve(module_name):
    module = importlib.import_module(module_name)

    for name in getattr(module, "__all__", []):
        assert getattr(module, name) is not None, f"{module_name}.{name} did not resolve"


@pytest.mark.parametrize(("mirror_module", "source_module"), MIRROR_MODULES)
def test_mirror_modules_match_backing_api_exports(mirror_module, source_module):
    assert exported_names(mirror_module) == exported_names(source_module)


def test_top_level_ptodsl_exports_import_cleanly():
    for name in ptodsl.__all__:
        assert getattr(ptodsl, name) is not None
