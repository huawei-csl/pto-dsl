Usage:

```bash
export PTODSL_TEST_DEVICE_ID=0
pytest -v .
```

## Test directories

| Directory | Needs NPU? | Needs ptoas CLI? | Needs bisheng? | What it tests |
|-----------|:----------:|:----------------:|:--------------:|---------------|
| `frontend/` | No | No (auto-skips) | No (auto-skips) | Python DSL bindings, IR generation, ptoas assembly, bisheng compilation |
| `npu/` | Yes (runtime) | Yes (build) | Yes (build) | On-device correctness, end-to-end |

`frontend/test_compile.py` auto-discovers every `test_*_ir.py` sibling and runs ptoas/bisheng on them.
Tests that need ptoas or bisheng are skipped automatically when the tools are not available.

Note: put all on-device tests in [npu](./npu) subdirectory.
If `PTODSL_TEST_DEVICE_ID` is not set, NPU tests default to `0` (resolved as `npu:0`) and print a warning.
