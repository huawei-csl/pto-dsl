#!/usr/bin/env python3
import os
import re
import shutil
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
    skipped = 0

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

            py_expr = m_py.group(1).strip()
            pto_expr = m_py.group(2).strip()

            py_match = py_path_re.search(py_expr)
            if not py_match:
                skipped += 1
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

            if not ptoas_line or not cpp_expr:
                skipped += 1
                continue

            py_rel = normalize_rel_path(expand_vars(py_match.group(1), variables))
            pto_rel = normalize_rel_path(expand_vars(pto_expr, variables))
            cpp_rel = normalize_rel_path(expand_vars(cpp_expr, variables))

            py_src = compile_path.parent / py_rel
            pto_src = compile_path.parent / pto_rel
            cpp_src = compile_path.parent / cpp_rel

            if not (py_src.exists() and pto_src.exists() and cpp_src.exists()):
                skipped += 1
                continue

            example_name = Path(py_rel).stem
            dst = unique_dir(out_dir / rel_dir / example_name)
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
                f"source_compile={compile_path}",
                f"source_dir={compile_path.parent}",
                "",
            ]
            (dst / "source_info.txt").write_text("\n".join(meta), encoding="utf-8")

            copied += 1

    print(f"Collected {copied} translation examples into {out_dir}")
    print(f"Skipped {skipped} entries (missing match or files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
