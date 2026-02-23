# Validate all examples on NPU

```bash
python ./validate_all_examples.py
```

## What it does

- Scans `examples/**/README.md` (excluding this top-level README).
- Reads the first fenced code block marked as `bash`, `sh`, `shell`, or unlabeled.
- Runs each command in that example's directory with `subprocess`.
- Continues running remaining examples even if one fails.
- Prints a pytest-like summary with `PASSED`/`FAILED` entries and failure details.
