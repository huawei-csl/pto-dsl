#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
from pathlib import Path


def strip_quotes(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def normalize_rel_path(token: str) -> str:
    token = strip_quotes(token).strip()
    if token.startswith("./"):
        token = token[2:]
    return token


def expand_vars(text: str, variables: dict[str, str]) -> str:
    text = text.strip()

    def repl_braced(match: re.Match[str]) -> str:
        name = match.group(1)
        return variables.get(name, match.group(0))

    def repl_plain(match: re.Match[str]) -> str:
        name = match.group(1)
        return variables.get(name, match.group(0))

    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl_braced, text)
    text = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", repl_plain, text)
    return text


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = Path(f"{base}_{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def parse_var_assignment(line: str) -> tuple[str, str] | None:
    m = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=(["\'])([^"\']+)\2\s*$', line)
    if not m:
        return None
    return m.group(1), m.group(3)


def main() -> int:
    repo_root = Path(os.environ["REPO_ROOT"]).resolve()
    aot_dir = Path(os.environ["AOT_DIR"]).resolve()
    out_dir = Path(os.environ["OUT_DIR"]).resolve()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compile_scripts = sorted(
        p for p in aot_dir.rglob("compile*.sh") if "example_translation" not in p.parts
    )

    py_to_pto_re = re.compile(r"^\s*python(?:3)?\s+(.+?)\s*>\s*(\S+\.pto)\s*$")
    py_path_re = re.compile(r"([A-Za-z0-9_./-]+\.py)\b")
    ptoas_re = re.compile(
        r"^\s*ptoas\b.*?(?P<pto>\S+\.pto)\s*(?:-o\s*(?P<cpp1>\S+\.cpp)|>\s*(?P<cpp2>\S+\.cpp))"
    )

    copied = 0
    failed = 0
    found = 0
    results: list[dict[str, str]] = []

    for compile_path in compile_scripts:
        rel_dir = compile_path.parent.relative_to(aot_dir)
        lines = compile_path.read_text(encoding="utf-8").splitlines()
        variables: dict[str, str] = {}

        for i, line in enumerate(lines):
            assign = parse_var_assignment(line)
            if assign:
                variables[assign[0]] = assign[1]

            m_py = py_to_pto_re.match(line)
            if not m_py:
                continue

            found += 1
            py_expr = m_py.group(1).strip()
            pto_expr = m_py.group(2).strip()

            py_match = py_path_re.search(py_expr)
            if not py_match:
                failed += 1
                results.append(
                    {
                        "name": f"{rel_dir}/line{i + 1}",
                        "status": "FAIL",
                        "reason": "python command does not contain a .py script path",
                    }
                )
                continue

            ptoas_line = None
            cpp_expr = None

            expanded_pto = normalize_rel_path(expand_vars(pto_expr, variables))
            for j in range(i + 1, len(lines)):
                m_as = ptoas_re.match(lines[j])
                if not m_as:
                    continue
                pto_in_ptoas = normalize_rel_path(expand_vars(m_as.group("pto"), variables))
                if pto_in_ptoas != expanded_pto:
                    continue
                ptoas_line = lines[j].strip()
                cpp_expr = (m_as.group("cpp1") or m_as.group("cpp2") or "").strip()
                break

            py_rel = normalize_rel_path(expand_vars(py_match.group(1), variables))
            pto_rel = normalize_rel_path(expand_vars(pto_expr, variables))
            example_name = f"{rel_dir}/{Path(py_rel).stem}:{Path(pto_rel).name}"

            if not ptoas_line or not cpp_expr:
                failed += 1
                results.append(
                    {
                        "name": example_name,
                        "status": "FAIL",
                        "reason": "no matching ptoas command found for generated .pto",
                    }
                )
                continue

            cpp_rel = normalize_rel_path(expand_vars(cpp_expr, variables))

            py_src = compile_path.parent / py_rel
            pto_src = compile_path.parent / pto_rel
            cpp_src = compile_path.parent / cpp_rel

            if not py_src.exists():
                failed += 1
                results.append(
                    {
                        "name": example_name,
                        "status": "FAIL",
                        "reason": f"python source does not exist: {py_src}",
                    }
                )
                continue

            run_env = os.environ.copy()
            for key, value in variables.items():
                run_env[key] = expand_vars(value, variables)

            py_cmd = line.strip()
            py_run = subprocess.run(
                py_cmd,
                shell=True,
                cwd=compile_path.parent,
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
                ptoas_line,
                shell=True,
                cwd=compile_path.parent,
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
                        "reason": f"ptoas command failed: {ptoas_line}" + (f" | {output}" if output else ""),
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

            dst = unique_dir(out_dir / rel_dir / Path(py_rel).stem)
            dst.mkdir(parents=True, exist_ok=True)

            shutil.copy2(py_src, dst / py_src.name)
            shutil.copy2(pto_src, dst / pto_src.name)
            shutil.copy2(cpp_src, dst / cpp_src.name)

            commands = [
                "#!/usr/bin/env bash",
                "set -e",
                line.strip(),
                ptoas_line,
                "",
            ]
            (dst / "compile.sh").write_text("\n".join(commands), encoding="utf-8")

            meta = [
                f"source_compile={compile_path.relative_to(repo_root)}",
                f"source_dir={compile_path.parent.relative_to(repo_root)}",
                "",
            ]
            (dst / "source_info.txt").write_text("\n".join(meta), encoding="utf-8")

            copied += 1
            results.append(
                {
                    "name": example_name,
                    "status": "OK",
                    "reason": f"collected to {dst}",
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
