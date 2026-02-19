# generic_dynamic_multicore tests

Tests for binary elementwise kernels (1D and 2D) across ops and dtypes.

## What the tests do

For each `(op, dtype)` combination:
1. **Compile** — `compile.sh` generates MLIR IR, runs `ptoas`, compiles with `bisheng`, produces a `.so`
2. **Build check** — asserts the `.so` was created
3. **1D precision** — runs the kernel on 1D shapes and checks against the PyTorch reference
4. **2D precision** — runs the kernel on 2D shapes and checks against the PyTorch reference

The `.so` is deleted after all tests for that combination finish.

## Run all tests

```bash
cd tests/npu/generic_dynamic_multicore
pytest --forked test_builder.py
```

## Run a specific op and dtype

```bash
pytest --forked test_builder.py -k "add-float32"
pytest --forked test_builder.py -k "mul-float16"
```

## Run only one test type

```bash
pytest --forked test_builder.py -k "test_binary_1d_precision"
pytest --forked test_builder.py -k "test_binary_2d_precision and add-float32"
```

## Ops and dtypes

| op  | float32 | float16 | int16 |
|-----|---------|---------|-------|
| add | ✓       | ✓       | ✓     |
| sub | ✓       | ✓       | ✓     |
| mul | ✓       | ✓       | ✓     |
| div | ✓       | ✓       | skip  |

## Compile a kernel manually

```bash
bash compile.sh <op_name> <dtype>
# e.g.
bash compile.sh add float32
bash compile.sh or int16
```

Output: `<op>_<dtype>_lib.so` in the same directory.

## Requirements

- `ptoas` and `bisheng` on `PATH`
- `/sources/pto-isa` present
- `torch_npu` installed