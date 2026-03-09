"""
Agentic kernel optimizer.

Uses Claude Opus with tool use to iteratively edit, build, test, and benchmark
PTO-DSL kernels until a faster-than-baseline implementation is found.

Usage:
    ANTHROPIC_API_KEY=sk-... python openenv/agent_search.py --config openenv/kernel_config.toml

The agent is given five tools that map 1-to-1 onto KernelSearchEnv actions:
    read_file    – inspect any file in the kernel directory
    edit_file    – apply a targeted text replacement to a source file
    build        – compile the current source to a shared library
    run_tests    – check correctness of the compiled kernel
    benchmark    – measure latency and compute speedup vs baseline

Claude decides what to change, builds, verifies, and benchmarks in a loop.
"""

import argparse
import shutil
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from openenv import KernelSearchEnv, KernelAction

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_env(cfg: dict) -> tuple["KernelSearchEnv", Path, Path]:
    """Returns (env, work_dir, ref_dir).

    ref_dir is kernel_dir (the reference/skeleton).  For create mode work_dir
    starts empty; for optimize mode it is a full copy of ref_dir.
    """
    ref_dir = Path(cfg["kernel_dir"])
    if not ref_dir.is_absolute():
        ref_dir = ROOT / ref_dir

    # Resolve working directory
    if "work_dir" in cfg:
        work_dir = Path(cfg["work_dir"])
        if not work_dir.is_absolute():
            work_dir = ROOT / work_dir
    else:
        default_name = cfg.get("kernel_name", ref_dir.name)
        work_dir = ROOT / "examples" / "agent" / default_name

    if work_dir.exists():
        shutil.rmtree(work_dir)

    if cfg.get("mode") == "create":
        # Start with an empty directory — agent creates all files from scratch
        work_dir.mkdir(parents=True)
        print(f"Working directory: {work_dir}  (empty — agent will create all files)")
    else:
        shutil.copytree(ref_dir, work_dir)
        print(f"Working directory: {work_dir}  (originals in {ref_dir} are untouched)")

    baseline_files = {
        name: (ref_dir / name).read_text(encoding="utf-8")
        for name in cfg.get("baseline_files", [])
    }

    def resolve_cmd(cmd: list[str]) -> list[str]:
        return [sys.executable if c == "python" else c for c in cmd]

    bench_cmd = resolve_cmd(cfg["bench_cmd"]) if "bench_cmd" in cfg else None

    env = KernelSearchEnv(
        repo_path=str(work_dir),
        build_cmd=resolve_cmd(cfg["build_cmd"]),
        test_cmd=resolve_cmd(cfg["test_cmd"]),
        bench_cmd=bench_cmd,
        baseline_files=baseline_files,
    )
    return env, work_dir, ref_dir


# ---------------------------------------------------------------------------
# Parse args and initialise globals
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="Agentic kernel optimizer")
_parser.add_argument(
    "--config", default=str(Path(__file__).parent / "kernel_config.toml"),
    help="Path to kernel TOML config (default: openenv/kernel_config.toml)",
)
_parser.add_argument("--max-turns", type=int, default=30)
_parser.add_argument("--max-tokens", type=int, default=None,
    help="Max tokens per API call (overrides config; default: 8192)")
_args = _parser.parse_args()

CFG = load_config(_args.config)
EXAMPLE_DIR: Path
REF_DIR: Path
env: KernelSearchEnv
env, EXAMPLE_DIR, REF_DIR = build_env(CFG)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read a source file from the kernel directory. "
            "Use this first to understand the current implementation before proposing changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the kernel directory, e.g. 'add_builder.py'",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace a unique snippet of text in a source file with new text. "
            "'old' must match exactly (including whitespace). "
            "Make one focused change at a time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path"},
                "old":  {"type": "string", "description": "Exact text to find and replace"},
                "new":  {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "build",
        "description": "Compile the current source files into a shared library (.so). Must be called after every edit before testing or benchmarking.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_tests",
        "description": "Run the correctness test suite. Returns pass/fail and any error output. Always call this after build before benchmark.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "benchmark",
        "description": "Measure the kernel latency and compute speedup vs the baseline. Returns latency_ms and speedup_vs_baseline.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "write_file",
        "description": (
            "Write the complete content of a file, creating it if it doesn't exist. "
            "Use this to write a new kernel file from scratch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Relative file path"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executor  (bridges Claude tool calls → KernelSearchEnv)
# ---------------------------------------------------------------------------
def execute_tool(name: str, tool_input: dict) -> str:
    if name == "read_file":
        path = EXAMPLE_DIR / tool_input["path"]
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: file not found: {tool_input['path']}"

    if name == "edit_file":
        for key in ("path", "old", "new"):
            if key not in tool_input:
                return f"Error: edit_file requires '{key}' parameter."
        obs = env.step(KernelAction("edit", {
            "path": tool_input["path"],
            "old":  tool_input["old"],
            "new":  tool_input["new"],
        }))
        return obs.summary

    if name == "build":
        obs = env.step(KernelAction("build", {}))
        return obs.summary

    if name == "run_tests":
        obs = env.step(KernelAction("test", {}))
        status = "PASS" if obs.passed_tests else "FAIL"
        return f"{status}\n{obs.summary}"

    if name == "benchmark":
        obs = env.step(KernelAction("benchmark", {}))
        if obs.latency_ms is None:
            return f"Benchmark failed: {obs.summary}"
        return (
            f"latency_ms={obs.latency_ms:.4f}  "
            f"speedup={obs.speedup_vs_baseline:.4f}x vs baseline ({env.baseline_ms:.4f} ms)\n"
            f"Best so far: {env.best_ms:.4f} ms ({env.best_speedup:.4f}x)"
        )

    if name == "write_file":
        for key in ("path", "content"):
            if key not in tool_input:
                return f"Error: write_file requires '{key}' parameter."
        obs = env.step(KernelAction("edit", {
            "path":    tool_input["path"],
            "content": tool_input["content"],
        }))
        return obs.summary

    return f"Unknown tool: {name}"

# ---------------------------------------------------------------------------
# System prompt  (built from config)
# ---------------------------------------------------------------------------
import textwrap as _textwrap


def _github_to_raw(url: str) -> str:
    """Convert a github.com blob URL to raw.githubusercontent.com."""
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def _fetch_url(url: str, max_chars: int = 40_000) -> str:
    url = _github_to_raw(url)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... (truncated at {max_chars} chars)"
        return content
    except Exception as exc:
        return f"(fetch failed: {exc})"


def _build_system_prompt(cfg: dict, kernel_dir: Path) -> str:
    name      = cfg.get("kernel_name", "kernel")
    main_file = cfg.get("main_file", "builder.py")
    tuning    = cfg.get("tuning_notes", "").strip()
    readonly  = cfg.get("readonly_files", [])

    readonly_rule = (
        f"- Never modify these files (test harness): {', '.join(readonly)}."
        if readonly else ""
    )

    files = sorted(f.name for f in kernel_dir.iterdir() if f.is_file())
    file_listing = ", ".join(files)

    # Embed preload_files verbatim so the agent doesn't waste turns reading them
    preload_section = ""
    for fname in cfg.get("preload_files", []):
        fpath = kernel_dir / fname
        try:
            content = fpath.read_text(encoding="utf-8")
            preload_section += f"\n### {fname}\n```\n{content}\n```\n"
        except FileNotFoundError:
            preload_section += f"\n### {fname}\n(file not found)\n"

    if preload_section:
        preload_section = "\n## Pre-loaded files (do NOT re-read these)\n" + preload_section

    # Embed local reference files (repo-relative paths)
    local_section = ""
    for fpath_str in cfg.get("local_context_files", []):
        fpath = ROOT / fpath_str
        try:
            content = fpath.read_text(encoding="utf-8")
            local_section += f"\n### {fpath.name}\n```python\n{content}\n```\n"
        except FileNotFoundError:
            local_section += f"\n### {fpath_str}\n(file not found)\n"
    if local_section:
        local_section = "\n## PTO-DSL API (local, authoritative)\n" + local_section

    # Fetch remote reference docs and embed them
    url_section = ""
    for entry in cfg.get("context_urls", []):
        if isinstance(entry, dict):
            url, label = entry["url"], entry.get("label", entry["url"])
        else:
            url, label = entry, entry
        print(f"  Fetching context: {label} …")
        content = _fetch_url(url)
        url_section += f"\n### {label}\n```\n{content}\n```\n"
    if url_section:
        url_section = "\n## Reference documentation (read-only, do not edit)\n" + url_section

    return _textwrap.dedent(f"""\
        You are an expert NPU kernel engineer specialising in Ascend/PTO-DSL performance tuning.

        Your task is to find a version of the {name} kernel that is FASTER than the baseline.

        The kernel directory contains exactly these files: {file_listing}
        Do NOT attempt to read any other filenames — they do not exist.
        {preload_section}{local_section}{url_section}
        ## Your workflow
        1. Propose one targeted change based on the pre-loaded files above
        2. edit_file(...)
        3. build()  — check for compile errors
        4. run_tests()  — ensure correctness
        5. benchmark()  — measure speedup

        Repeat until speedup > 1.0 (faster than baseline) or you have exhausted ideas.

        ## Key tuning notes
        {tuning}

        ## Rules
        {readonly_rule}
        - One edit at a time; build and verify after each change.
        - If a build or test fails, revert the change with another edit_file call
          and try a different approach.
        - Stop as soon as you achieve a confirmed speedup > 1.0x and explain what worked.
    """)

def _build_create_system_prompt(cfg: dict, work_dir: Path, ref_dir: Path) -> str:
    name        = cfg.get("kernel_name", "kernel")
    create_file = cfg.get("create_file", "builder.py")
    description = cfg.get("create_prompt", "").strip()

    files = sorted(f.name for f in work_dir.iterdir() if f.is_file())
    file_listing = ", ".join(files) if files else "(empty — you must create all files)"

    # Preload_files come from ref_dir (the reference skeleton), not the empty work_dir
    preload_section = ""
    for fname in cfg.get("preload_files", []):
        fpath = ref_dir / fname
        try:
            content = fpath.read_text(encoding="utf-8")
            preload_section += f"\n### {fname} (from reference kernel — adapt for {name})\n```\n{content}\n```\n"
        except FileNotFoundError:
            preload_section += f"\n### {fname}\n(file not found in reference)\n"
    if preload_section:
        preload_section = "\n## Reference files (adapt these for the new kernel — do NOT copy verbatim)\n" + preload_section

    # Embed local reference files (repo-relative paths)
    local_section = ""
    for fpath_str in cfg.get("local_context_files", []):
        fpath = ROOT / fpath_str
        try:
            content = fpath.read_text(encoding="utf-8")
            local_section += f"\n### {fpath.name}\n```python\n{content}\n```\n"
        except FileNotFoundError:
            local_section += f"\n### {fpath_str}\n(file not found)\n"
    if local_section:
        local_section = "\n## PTO-DSL API (local, authoritative)\n" + local_section

    url_section = ""
    for entry in cfg.get("context_urls", []):
        if isinstance(entry, dict):
            url, label = entry["url"], entry.get("label", entry["url"])
        else:
            url, label = entry, entry
        print(f"  Fetching context: {label} …")
        content = _fetch_url(url)
        url_section += f"\n### {label}\n```\n{content}\n```\n"
    if url_section:
        url_section = "\n## Reference documentation\n" + url_section

    return _textwrap.dedent(f"""\
        You are an expert NPU kernel engineer specialising in Ascend/PTO-DSL.

        Your task is to implement the {name} kernel entirely from scratch.
        The working directory starts with: {file_listing}
        You must create ALL required files yourself using write_file.

        {preload_section}{local_section}{url_section}
        ## Kernel to implement
        {description}

        ## Files you must create
        - `{create_file}`  — PTO-DSL kernel builder (the main implementation)
        - `compile.sh`     — build script (adapt from reference above; use {name}-specific names)
        - `caller.cpp`     — C++ entry point (adapt signature to match the {name} kernel exactly)
        - `run_{name}.py`  — correctness test script (adapt from reference; use `{name}_lib.so`)

        ## Your workflow
        1. Write all required files using write_file(path="...", content="...")
        2. build()  — check for compile errors; fix with write_file/edit_file as needed
        3. run_tests()  — verify correctness; fix failures and rebuild
        4. Once tests pass, call benchmark() to measure latency

        ## Rules
        - Study the reference files above carefully to understand the PTO-DSL API and build system.
        - The compile.sh and caller.cpp you create must be consistent with each other and with `{create_file}`.
        - If build or tests fail, fix the issue — do not give up.
        - Stop once tests pass and you have a benchmark result.
    """)


SYSTEM = (
    _build_create_system_prompt(CFG, EXAMPLE_DIR, REF_DIR)
    if CFG.get("mode") == "create"
    else _build_system_prompt(CFG, EXAMPLE_DIR)
)

# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------
def run_agent(max_turns: int = 30) -> None:
    mode = CFG.get("mode", "optimize")
    client = anthropic.Anthropic()

    if mode == "create":
        name = CFG.get("kernel_name", "kernel")
        create_file = CFG.get("create_file", "builder.py")
        print(f"Create mode — implementing {name} kernel from scratch …\n")
        messages: list[dict] = [{"role": "user", "content":
            f"Please implement the {name} kernel from scratch. "
            f"Create all required files (compile.sh, caller.cpp, {create_file}, run_{name}.py) "
            "using write_file. The reference files and PTO-DSL docs are in your system context above."}]
    else:
        print("Establishing baseline …")
        obs = env.reset()
        print(f"Baseline: {obs.latency_ms:.4f} ms\n")
        preloaded = CFG.get("preload_files", [])
        start_hint = (
            "The source files are already in your context above — start proposing a change directly."
            if preloaded else f"Start by reading {CFG.get('main_file', 'builder.py')}."
        )
        messages = [{"role": "user", "content":
            f"The baseline kernel latency is {obs.latency_ms:.4f} ms. "
            f"Please find a faster implementation. {start_hint}"}]

    tests_passed = False
    turn = 0
    while turn < max_turns:
        turn += 1
        print(f"\n{'─'*60}")
        print(f"Turn {turn}/{max_turns}")
        print(f"{'─'*60}")

        max_tokens = _args.max_tokens or CFG.get("max_tokens", 8192)
        for attempt in range(5):
            try:
                with client.messages.stream(
                    model="claude-opus-4-6",
                    max_tokens=max_tokens,
                    thinking={"type": "adaptive"},
                    system=SYSTEM,
                    tools=TOOLS,
                    messages=messages,
                ) as stream:
                    response = stream.get_final_message()
                break
            except anthropic.RateLimitError as e:
                wait = 60 * (attempt + 1)
                print(f"\n[Rate limit] Waiting {wait}s before retry ({attempt+1}/5)… ({e})")
                time.sleep(wait)
        else:
            raise RuntimeError("Rate limit retries exhausted")

        # Print any text Claude produced
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\n[Claude] {block.text.strip()}")

        # Append assistant turn — strip trailing thinking blocks (API rejects them as final block)
        content = list(response.content)
        while content and getattr(content[-1], "type", None) == "thinking":
            content.pop()
        if content:
            messages.append({"role": "assistant", "content": content})

        # Done if no tool calls
        if response.stop_reason == "end_turn":
            print("\n[Agent] Claude finished.")
            break

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"\n  → {block.name}({_fmt_input(block.input)})")
            result = execute_tool(block.name, block.input)
            print(f"  ← {result[:300]}" + ("…" if len(result) > 300 else ""))

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
            # Track test results for create-mode exit condition
            if block.name == "run_tests" and result.startswith("PASS"):
                tests_passed = True

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if mode == "create" and tests_passed:
            print("\n[Agent] Tests passed — kernel implemented successfully.")
            break

        if mode == "optimize" and env.best_speedup is not None and env.best_speedup > 1.0:
            print(f"\n[Agent] Speedup achieved: {env.best_speedup:.4f}x — stopping early.")
            break

    # Final summary
    print(f"\n{'='*60}")
    if mode == "create":
        print("CREATE COMPLETE")
        print(f"{'='*60}")
        print(f"  Result: {'tests passed' if tests_passed else 'did not reach passing tests'}")
        print(f"  Output: {EXAMPLE_DIR}")
    else:
        print("SEARCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Baseline:     {env.baseline_ms:.4f} ms")
        print(f"  Best found:   {env.best_ms:.4f} ms")
        print(f"  Best speedup: {env.best_speedup:.4f}x" if env.best_speedup else "  No improvement found.")


def _fmt_input(inp: dict) -> str:
    """Compact one-line representation of tool input for logging."""
    parts = []
    for k, v in inp.items():
        s = str(v).replace("\n", "\\n")
        parts.append(f"{k}={s[:60]!r}" if len(s) > 60 else f"{k}={s!r}")
    return ", ".join(parts)


if __name__ == "__main__":
    run_agent(max_turns=_args.max_turns)
