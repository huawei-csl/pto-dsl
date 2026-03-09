# openenv — Agentic Kernel Optimizer

Uses Claude Opus with OpenEnv to iteratively edit, build, test, and benchmark PTO-DSL kernels until a faster-than-baseline implementation is found.

## Installation

```bash
pip install openenv-core   
```

## Quick start

```bash
ANTHROPIC_API_KEY=sk-... python openenv/agent_search.py --config openenv/hadamard_config.toml
```

The agent will:
1. Copy the kernel directory to a `_opt/` working directory (originals untouched)
2. Preload source files and reference docs into its context
3. Iteratively edit → build → test → benchmark in a loop
4. Stop when it achieves a speedup > 1.0x or exhausts its turn budget

The optimized result is left in the `_opt/` directory.

## CLI options

```
python openenv/agent_search.py
  --config PATH       Path to a kernel TOML config (default: openenv/hadamard_config.toml)
  --max-turns N       Max agent turns (default: 30)
  --max-tokens N      Max tokens per API call (default: 8192; overrides config)
```

## Config file reference

Each kernel needs its own `.toml` config. See `hadamard_config.toml` or `geglu_config.toml` for complete examples.

```toml
# --- Required ---

# Path to the kernel source directory (relative to repo root, or absolute)
kernel_dir = "examples/aot/my_kernel"

# Shell commands run from inside the working directory
build_cmd = ["bash", "compile.sh"]
test_cmd  = ["python", "run_my_kernel.py"]
bench_cmd = ["python", "_bench_wrapper.py"]   # must print "latency_ms=<float>"

# Files snapshotted as baseline; restored on reset/revert
baseline_files = ["my_kernel_builder.py"]

# --- Agent context ---

# Primary file the agent is told to look at first
main_file = "my_kernel_builder.py"

# Human-readable kernel name used in prompts
kernel_name = "my_kernel"

# Free-form domain notes injected into the system prompt
tuning_notes = """
- Key constraints, valid parameter ranges, hardware limits, etc.
"""

# Files the agent must NOT modify (enforced via system prompt)
readonly_files = ["run_my_kernel.py", "caller.cpp"]

# Files embedded verbatim in the system prompt at startup (saves read_file turns)
preload_files = ["my_kernel_builder.py", "compile.sh", "caller.cpp"]

# Remote docs fetched at startup and embedded in the system prompt.
# GitHub blob URLs are auto-converted to raw.githubusercontent.com.
[[context_urls]]
url   = "https://github.com/org/repo/blob/main/some_reference.td"
label = "Human-readable label shown in prompt"

# --- Optional ---

# Working directory for edits. Defaults to "{kernel_dir}_opt".
# work_dir = "examples/aot/my_kernel_opt"

# Max tokens per API call (also settable via --max-tokens CLI flag)
# max_tokens = 8192
```

## Adding a new kernel

1. Make sure the kernel directory has:
   - A build script (e.g. `compile.sh`)
   - A test script (e.g. `run_my_kernel.py`) that exits 0 on pass
   - A bench wrapper `_bench_wrapper.py` that prints `latency_ms=<float>` to stdout

2. Create a config file (copy `hadamard_config.toml` and adjust paths/notes)

3. Run:
   ```bash
   ANTHROPIC_API_KEY=sk-... python openenv/agent_search.py --config openenv/my_kernel_config.toml
   ```

### Writing `_bench_wrapper.py`

The bench wrapper is a plain Python script executed as a subprocess. It must print exactly one line of the form:

```
latency_ms=0.1234
```

See `examples/aot/fast_hadamard/_bench_wrapper.py` or `examples/aot/geglu_dynamic_multicore/_bench_wrapper.py` for complete examples. Typical structure:

```python
# load lib, allocate tensors
# warmup loop
# timed loop with per-iteration events
ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / ITERS
print(f"latency_ms={ms:.4f}")
```

## Files

| File | Purpose |
|---|---|
| `agent_search.py` | Main entry point — agentic loop, config loading, tool executor |
| `kernel_opt_env.py` | `KernelSearchEnv` — edit/build/test/benchmark environment |
| `hadamard_config.toml` | Config for `examples/aot/fast_hadamard` |
| `geglu_config.toml` | Config for `examples/aot/geglu_dynamic_multicore` |
