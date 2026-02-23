# Validate all examples

Use `validate_all_examples.py` to execute each example using the command block in
that example's `README.md`.

## What it does

- Scans `examples/**/README.md` (excluding this top-level README).
- Reads the first fenced code block marked as `bash`, `sh`, `shell`, or unlabeled.
- Runs each command in that example's directory with `subprocess`.
- Continues running remaining examples even if one fails.
- Prints a pytest-like summary with `PASSED`/`FAILED` entries and failure details.

## Run

From the `examples` directory:

```bash
python ./validate_all_examples.py
```

Or from repository root:

```bash
python ./examples/validate_all_examples.py
```
