#!/usr/bin/env python3
"""Run all examples by executing commands from each example README."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


README_NAME = "README.md"
SUPPORTED_LANGS = {"bash", "sh", "shell", ""}


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass
class ExampleResult:
    name: str
    readme_path: Path
    commands: list[str]
    command_results: list[CommandResult]
    status: str
    elapsed_seconds: float = 0.0
    error: str = ""


def discover_example_readmes(examples_root: Path) -> list[Path]:
    readmes = [p for p in examples_root.rglob(README_NAME) if p.parent != examples_root]
    return sorted(readmes)


def extract_commands(readme_path: Path) -> list[str]:
    content = readme_path.read_text(encoding="utf-8")
    fenced_blocks = re.findall(r"```([^\n`]*)\n(.*?)```", content, flags=re.DOTALL)

    for lang, block in fenced_blocks:
        normalized_lang = lang.strip().lower()
        if normalized_lang not in SUPPORTED_LANGS:
            continue

        commands = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("$ "):
                line = line[2:].strip()
            commands.append(line)

        if commands:
            return commands

    return []


def run_example(example_name: str, readme_path: Path, commands: list[str]) -> ExampleResult:
    example_start = time.time()
    if not commands:
        return ExampleResult(
            name=example_name,
            readme_path=readme_path,
            commands=commands,
            command_results=[],
            status="FAILED",
            elapsed_seconds=time.time() - example_start,
            error="No runnable commands found in README fenced code blocks.",
        )

    command_results: list[CommandResult] = []
    cwd = readme_path.parent

    for command in commands:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            text=True,
            capture_output=True,
        )
        command_results.append(
            CommandResult(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        )
        if completed.returncode != 0:
            return ExampleResult(
                name=example_name,
                readme_path=readme_path,
                commands=commands,
                command_results=command_results,
                status="FAILED",
                elapsed_seconds=time.time() - example_start,
                error=f"Command failed: {command}",
            )

    return ExampleResult(
        name=example_name,
        readme_path=readme_path,
        commands=commands,
        command_results=command_results,
        status="PASSED",
        elapsed_seconds=time.time() - example_start,
    )


def print_header(total: int) -> None:
    print("=" * 78)
    print("example session starts")
    print(f"collected {total} example(s)")
    print("=" * 78)


def print_failure_details(result: ExampleResult) -> None:
    print(f"{result.name}")
    print(f"  README: {result.readme_path}")
    if result.error:
        print(f"  Error: {result.error}")

    for command_result in result.command_results:
        if command_result.returncode == 0:
            continue
        print(f"  Failed command: {command_result.command}")
        print(f"  Exit code: {command_result.returncode}")
        if command_result.stdout.strip():
            print("  stdout:")
            for line in command_result.stdout.rstrip().splitlines():
                print(f"    {line}")
        if command_result.stderr.strip():
            print("  stderr:")
            for line in command_result.stderr.rstrip().splitlines():
                print(f"    {line}")
        break


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate all examples by running commands from each example README."
    )
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Examples root directory. Defaults to this script directory.",
    )
    args = parser.parse_args()

    examples_root = args.root.resolve()
    readmes = discover_example_readmes(examples_root)
    print_header(len(readmes))

    results: list[ExampleResult] = []
    start = time.time()

    for readme in readmes:
        example_name = readme.parent.relative_to(examples_root).as_posix()
        commands = extract_commands(readme)
        result = run_example(example_name, readme, commands)
        results.append(result)
        print(f"{result.name:<48} {result.status:<7} [{result.elapsed_seconds:.2f}s]")

    elapsed = time.time() - start
    passed = [r for r in results if r.status == "PASSED"]
    failed = [r for r in results if r.status == "FAILED"]

    print()
    print("=" * 78)
    print("short summary info")
    print("=" * 78)
    for result in failed:
        print(f"FAILED {result.name} [{result.elapsed_seconds:.2f}s]")
    for result in passed:
        print(f"PASSED {result.name} [{result.elapsed_seconds:.2f}s]")
    print("=" * 78)
    print(f"{len(passed)} passed, {len(failed)} failed in {elapsed:.2f}s")

    if failed:
        print()
        print("=" * 78)
        print("failure details")
        print("=" * 78)
        for result in failed:
            print_failure_details(result)
            print("-" * 78)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
