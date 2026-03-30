#!/usr/bin/env python3
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "tests" / "api"))

    from _contract import SNAPSHOT_PATH, collect_public_api_snapshot, snapshot_json

    SNAPSHOT_PATH.write_text(
        snapshot_json(collect_public_api_snapshot()),
        encoding="utf-8",
    )
    print(f"Updated {SNAPSHOT_PATH.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
