Usage:

```bash
export PTODSL_TEST_DEVICE_ID=0
pytest -v .
```

Note: put all on-device tests in [npu](./npu) subdirectory. The other directories only need host CPU.
If `PTODSL_TEST_DEVICE_ID` is not set, NPU tests default to `0` (resolved as `npu:0`) and print a warning.
