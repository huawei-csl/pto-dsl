#!/usr/bin/env python3
"""Collect python->pto->cpp translation examples into a reference directory.

Usage:
  python collect_example_translate.py
  python collect_example_translate.py --aot-dir /path/to/examples/aot --out-dir /tmp/example_translation
"""
import json
import argparse
import os
import shutil
import subprocess
from pathlib import Path

def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = Path(f"{base}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


REQUIRED_FIELDS = {
    "example_dir",
    "compile_script",
    "py_source",
    "py_command",
    "ptoas_command",
    "pto_file",
    "cpp_file",
}
OPTIONAL_FIELDS = {"dependency"}


def load_example_list(config_path: Path) -> list[dict[str, object]]:
    if not config_path.exists():
        raise FileNotFoundError(f"example config not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("example config root must be a list")

    examples: list[dict[str, object]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"entry #{idx} must be an object")
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            raise ValueError(f"entry #{idx} missing fields: {sorted(missing)}")
        unknown = set(item.keys()) - REQUIRED_FIELDS - OPTIONAL_FIELDS
        if unknown:
            raise ValueError(f"entry #{idx} has unknown fields: {sorted(unknown)}")

        normalized: dict[str, str | list[str]] = {}
        for key in REQUIRED_FIELDS:
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"entry #{idx} field '{key}' must be a non-empty string")
            normalized[key] = value

        dependency = item.get("dependency", [])
        if not isinstance(dependency, list):
            raise ValueError(f"entry #{idx} field 'dependency' must be a list of strings")
        dep_list: list[str] = []
        for dep_idx, dep in enumerate(dependency):
            if not isinstance(dep, str) or not dep.strip():
                raise ValueError(
                    f"entry #{idx} dependency[{dep_idx}] must be a non-empty string"
                )
            dep_list.append(dep)
        normalized["dependency"] = dep_list
        examples.append(normalized)
    return examples


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_repo_root = (script_dir / "../../../..").resolve()
    parser = argparse.ArgumentParser(description="Collect python->pto->cpp translation examples.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help="Repository root path (default: script_dir/../../../..).",
    )
    parser.add_argument(
        "--aot-dir",
        type=Path,
        default=default_repo_root / "examples/aot",
        help="AOT examples directory (default: <repo-root>/examples/aot).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(script_dir / "../references/example_translation").resolve(),
        help="Output directory (default: script_dir/../references/example_translation).",
    )
    parser.add_argument(
        "--example-config",
        type=Path,
        default=script_dir / "example_list.json",
        help="Example list json path (default: script_dir/example_list.json).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    aot_dir = args.aot_dir.resolve()
    out_dir = args.out_dir.resolve()
    example_config = args.example_config.resolve()
    example_list = load_example_list(example_config)

    if not aot_dir.is_dir():
        raise FileNotFoundError(f"AOT examples directory not found: {aot_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    failed = 0
    found = len(example_list)
    results: list[dict[str, str]] = []

    for idx, example in enumerate(example_list, start=1):
        rel_dir = Path(str(example["example_dir"]))
        example_dir = aot_dir / rel_dir
        py_rel = Path(str(example["py_source"]))
        py_source = example_dir / py_rel
        py_cmd = str(example["py_command"])
        ptoas_cmd = str(example["ptoas_command"])
        example_name = f"{example['example_dir']}:{example['pto_file']}"
        progress_name = py_rel.stem
        dependencies = example.get("dependency", [])
        print(f"[{idx}/{found}] collecting {progress_name}")

        if not py_source.exists():
            failed += 1
            results.append(
                {
                    "name": example_name,
                    "status": "FAIL",
                    "reason": f"python source does not exist: {py_source}",
                }
            )
            continue

        dst = unique_dir(out_dir / rel_dir / Path(str(example["pto_file"])).stem)
        dst.mkdir(parents=True, exist_ok=True)

        py_dst = dst / py_rel
        py_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(py_source, py_dst)
        dep_copy_failed = False
        for dep in dependencies:
            dep_src = example_dir / dep
            if not dep_src.exists():
                failed += 1
                results.append(
                    {
                        "name": example_name,
                        "status": "FAIL",
                        "reason": f"dependency does not exist: {dep_src}",
                    }
                )
                dep_copy_failed = True
                break
            dep_dst = dst / dep
            dep_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dep_src, dep_dst)
        if dep_copy_failed:
            continue

        run_env = os.environ.copy()

        py_run = subprocess.run(
            py_cmd,
            shell=True,
            cwd=dst,
            env=run_env,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if py_run.returncode != 0:
            failed += 1
            output = (py_run.stdout or "").strip()
            results.append(
                {
                    "name": example_name,
                    "status": "FAIL",
                    "reason": f"python command failed: {py_cmd}" + (f" | {output}" if output else ""),
                }
            )
            continue

        ptoas_run = subprocess.run(
            ptoas_cmd,
            shell=True,
            cwd=dst,
            env=run_env,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if ptoas_run.returncode != 0:
            failed += 1
            output = (ptoas_run.stdout or "").strip()
            results.append(
                {
                    "name": example_name,
                    "status": "FAIL",
                    "reason": f"ptoas command failed: {ptoas_cmd}" + (f" | {output}" if output else ""),
                }
            )
            continue

        pto_dst = dst / str(example["pto_file"])
        cpp_dst = dst / str(example["cpp_file"])
        if not (pto_dst.exists() and cpp_dst.exists()):
            failed += 1
            results.append(
                {
                    "name": example_name,
                    "status": "FAIL",
                    "reason": (
                        "expected outputs missing after compile: "
                        f"{example['pto_file']}, {example['cpp_file']}"
                    ),
                }
            )
            continue

        commands = [
            "#!/usr/bin/env bash",
            "set -e",
            py_cmd,
            ptoas_cmd,
            "",
        ]
        (dst / "compile.sh").write_text("\n".join(commands), encoding="utf-8")

        copied += 1
        results.append(
            {
                "name": example_name,
                "status": "OK",
                "reason": f"collected to {dst.relative_to(repo_root)}",
            }
        )

    print(f"Discovered {found} python->pto candidates under {aot_dir}")
    for item in results:
        print(f"[{item['status']}] {item['name']} - {item['reason']}")
    print(f"Collected {copied} translation examples into {out_dir}")
    print(f"Failed to collect {failed} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
