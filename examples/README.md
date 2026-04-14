# Validate all examples on NPU

Set the target NPU device with `PTODSL_TEST_DEVICE_ID` (for example `3`).
If this environment variable is not set, example/test scripts default to `0` (resolved as `npu:0`) and print a warning.

JIT (AOT) examples also require `PTO_LIB_PATH` to point to the `pto-isa` installation so the compiler can find its headers.
By default it resolves to `/sources/pto-isa`, but you can export it to a different path before running.

```bash
export PTODSL_TEST_DEVICE_ID=0
export PTO_LIB_PATH=/sources/pto-isa  # optional: override the default path
python ./validate_all_examples.py
```

## What it does

- Scans `examples/**/README.md` (excluding this top-level README).
- Reads the first fenced code block marked as `bash`, `sh`, `shell`, or unlabeled.
- Runs each command in that example's directory with `subprocess`.
- Continues running remaining examples even if one fails.
- Prints a pytest-like summary with `PASSED`/`FAILED` entries and failure details.
