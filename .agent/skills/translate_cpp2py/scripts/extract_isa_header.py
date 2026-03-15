#!/usr/bin/env python3
"""
Extract signature-only C++ headers for API documentation.

This script copies all .hpp files from a source tree into a target tree and
strips function implementation bodies, keeping declarations/signatures only.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CONTROL_KEYWORDS = {"if", "for", "while", "switch", "catch"}


def is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def prev_non_ws(text: str, i: int) -> int:
    while i >= 0 and text[i].isspace():
        i -= 1
    return i


def get_prev_word(text: str, i: int) -> str:
    i = prev_non_ws(text, i)
    if i < 0:
        return ""
    end = i
    while i >= 0 and is_ident_char(text[i]):
        i -= 1
    return text[i + 1 : end + 1]


def looks_like_function_body(text: str, brace_idx: int) -> bool:
    # Find nearest ')' before '{', then trace back to its matching '('.
    j = prev_non_ws(text, brace_idx - 1)
    if j < 0:
        return False

    paren = 0
    found_rparen = False
    while j >= 0:
        ch = text[j]
        if ch == ")":
            paren += 1
            found_rparen = True
        elif ch == "(":
            if paren == 0:
                return False
            paren -= 1
            if paren == 0:
                break
        j -= 1

    if not found_rparen or j < 0:
        return False

    lparen_idx = j

    # Filter out control flow blocks: if (...) { ... }, for (...) { ... }, etc.
    keyword = get_prev_word(text, lparen_idx - 1)
    if keyword in CONTROL_KEYWORDS:
        return False

    stmt_start = max(
        text.rfind(";", 0, lparen_idx),
        text.rfind("{", 0, lparen_idx),
        text.rfind("}", 0, lparen_idx),
    )
    chunk = text[stmt_start + 1 : brace_idx]
    chunk = re.sub(r"/\*.*?\*/", " ", chunk, flags=re.S)
    chunk = re.sub(r"//.*", " ", chunk)
    chunk = chunk.strip()

    # Avoid touching type/object blocks.
    if re.search(r"\b(class|struct|namespace|enum|union)\b", chunk):
        return False

    # Must still look like a callable signature.
    return "(" in chunk and ")" in chunk


def skip_string(text: str, i: int) -> int:
    quote = text[i]
    i += 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == quote:
            return i + 1
        i += 1
    return i


def strip_function_bodies(text: str) -> str:
    i = 0
    n = len(text)
    out: list[str] = []

    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            if j == -1:
                out.append(text[i:])
                break
            out.append(text[i : j + 1])
            i = j + 1
            continue

        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j == -1:
                out.append(text[i:])
                break
            out.append(text[i : j + 2])
            i = j + 2
            continue

        ch = text[i]
        if ch in {'"', "'"}:
            j = skip_string(text, i)
            out.append(text[i:j])
            i = j
            continue

        if ch == "{" and looks_like_function_body(text, i):
            # Replace definition body with declaration terminator.
            out.append(";")
            i += 1
            depth = 1
            while i < n and depth > 0:
                if text.startswith("//", i):
                    j = text.find("\n", i)
                    if j == -1:
                        i = n
                        break
                    i = j + 1
                    continue
                if text.startswith("/*", i):
                    j = text.find("*/", i + 2)
                    if j == -1:
                        i = n
                        break
                    i = j + 2
                    continue

                ch2 = text[i]
                if ch2 in {'"', "'"}:
                    i = skip_string(text, i)
                    continue

                if ch2 == "{":
                    depth += 1
                elif ch2 == "}":
                    depth -= 1
                i += 1

            while i < n and text[i] in " \t":
                i += 1
            if i < n and text[i] == ";":
                i += 1
            continue

        out.append(ch)
        i += 1

    cleaned = "".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def derive_default_dst(src: Path) -> Path:
    if src.name.endswith("_full"):
        return src.with_name(src.name[: -len("_full")] + "_header")
    return src.with_name(src.name + "_header")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract signature-only .hpp headers from a source tree."
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Source directory containing .hpp files (e.g. a2a3_full).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        help="Destination directory. Default: sibling *_header directory.",
    )
    parser.add_argument(
        "--glob",
        default="*.hpp",
        help="File pattern under source root (default: *.hpp).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_root = args.src.resolve()
    dst_root = args.dst.resolve() if args.dst else derive_default_dst(src_root)

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(f"Source directory does not exist: {src_root}")

    files = sorted(src_root.rglob(args.glob))
    if not files:
        raise SystemExit(f"No files matched '{args.glob}' under: {src_root}")

    count = 0
    for src in files:
        if src.suffix != ".hpp":
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        content = src.read_text(encoding="utf-8")
        stripped = strip_function_bodies(content)
        dst.write_text(stripped, encoding="utf-8")
        count += 1

    print(f"Generated {count} files")
    print(f"src: {src_root}")
    print(f"dst: {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
