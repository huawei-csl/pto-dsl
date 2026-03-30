import importlib
import json
from pathlib import Path


PUBLIC_MODULES = [
    "ptodsl",
    "ptodsl.pto",
    "ptodsl.tile",
    "ptodsl.scalar",
    "ptodsl.micro",
    "ptodsl.compiler",
    "ptodsl.api.pto",
    "ptodsl.api.tile",
    "ptodsl.api.scalar",
    "ptodsl.api.micro",
    "ptodsl.api.control_flow",
    "ptodsl.api.synchronization",
    "ptodsl.api.type_def",
    "ptodsl.bench",
]

MIRROR_MODULES = [
    ("ptodsl.pto", "ptodsl.api.pto"),
    ("ptodsl.tile", "ptodsl.api.tile"),
    ("ptodsl.scalar", "ptodsl.api.scalar"),
    ("ptodsl.micro", "ptodsl.api.micro"),
]

SNAPSHOT_PATH = Path(__file__).with_name("public_api_snapshot.json")


def exported_names(module_name):
    module = importlib.import_module(module_name)
    exports = getattr(module, "__all__", None)
    if exports is None:
        raise AssertionError(f"{module_name} is missing __all__.")
    return sorted(dict.fromkeys(exports))


def collect_public_api_snapshot():
    return {module_name: exported_names(module_name) for module_name in PUBLIC_MODULES}


def load_snapshot():
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def snapshot_json(snapshot):
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
