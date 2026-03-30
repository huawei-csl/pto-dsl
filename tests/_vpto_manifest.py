import json
import os
from pathlib import Path


ENV_VAR = "PTOAS_VPTO_MANIFEST"


def vpto_manifest_path():
    repo_root = Path(__file__).resolve().parents[1]
    configured = os.environ.get(ENV_VAR)

    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        [
            repo_root.parent / "PTOAS" / "docs" / "vpto-manifest.json",
            repo_root / "PTOAS" / "docs" / "vpto-manifest.json",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not locate PTOAS vPTO manifest. Set "
        f"`{ENV_VAR}` or provide a sibling PTOAS checkout.\n"
        f"Searched:\n{searched}"
    )


def load_vpto_manifest():
    return json.loads(vpto_manifest_path().read_text(encoding="utf-8"))
