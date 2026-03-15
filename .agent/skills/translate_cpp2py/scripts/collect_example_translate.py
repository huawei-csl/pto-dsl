#!/usr/bin/env python3
import json
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


def load_example_list(config_path: Path) -> list[dict[str, str]]:
    if not config_path.exists():
        raise FileNotFoundError(f"example config not found: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("example config root must be a list")

    examples: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"entry #{idx} must be an object")
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            raise ValueError(f"entry #{idx} missing fields: {sorted(missing)}")

        normalized: dict[str, str] = {}
        for key in REQUIRED_FIELDS:
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"entry #{idx} field '{key}' must be a non-empty string")
            normalized[key] = value
        examples.append(normalized)
    return examples


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    example_config = script_dir / "example_list.json"
    repo_root = Path(os.environ["REPO_ROOT"]).resolve()
    aot_dir = Path(os.environ["AOT_DIR"]).resolve()
    out_dir = Path(os.environ["OUT_DIR"]).resolve()
    example_list = load_example_list(example_config)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    failed = 0
    found = len(example_list)
    results: list[dict[str, str]] = []

    for example in example_list:
        rel_dir = Path(example["example_dir"])
        example_dir = aot_dir / rel_dir
        py_source = example_dir / example["py_source"]
        pto_src = example_dir / example["pto_file"]
        cpp_src = example_dir / example["cpp_file"]
        py_cmd = example["py_command"]
        ptoas_cmd = example["ptoas_command"]
        example_name = f"{example['example_dir']}:{example['pto_file']}"

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

        run_env = os.environ.copy()

        py_run = subprocess.run(
            py_cmd,
            shell=True,
            cwd=example_dir,
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
            cwd=example_dir,
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

        if not (pto_src.exists() and cpp_src.exists()):
            failed += 1
            results.append(
                {
                    "name": example_name,
                    "status": "FAIL",
                    "reason": f"expected outputs missing after compile: {pto_src.name}, {cpp_src.name}",
                }
            )
            continue

        dst = unique_dir(out_dir / rel_dir / Path(example["pto_file"]).stem)
        dst.mkdir(parents=True, exist_ok=True)

        shutil.copy2(py_source, dst / py_source.name)
        shutil.copy2(pto_src, dst / pto_src.name)
        shutil.copy2(cpp_src, dst / cpp_src.name)

        commands = [
            "#!/usr/bin/env bash",
            "set -e",
            py_cmd,
            ptoas_cmd,
            "",
        ]
        (dst / "compile.sh").write_text("\n".join(commands), encoding="utf-8")

        meta = [
            f"source_compile={Path('examples/aot') / rel_dir / example['compile_script']}",
            f"source_dir={Path('examples/aot') / rel_dir}",
            "",
        ]
        (dst / "source_info.txt").write_text("\n".join(meta), encoding="utf-8")

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
