# Cube To Vector `TPUSH`/`TPOP` Example

This example wraps `refs/tpushpop_cv.cpp` into the same `compile.sh` + Python runner flow used by the AOT examples.

The kernel does:

- cube-side `TMATMUL`
- `TPUSH` from cube to vector
- vector-side `TPOP`
- vector-side bias add

## Run

```bash
python run_tpushpop_cv.py
```

That will:

1. call `compile.sh`
2. build `./tpushpop_cv_lib.so`
3. launch the kernel on NPU
4. compare against `A @ B + bias`

The wrapper fetches the runtime FFTS/control address inside `caller.cpp` with `rtGetC2cCtrlAddr(...)`, so the Python side only needs to provide the kernel inputs, output, and FIFO backing memory.

If your environment needs different PTO include roots:

```bash
PTO_INCLUDE_PATH=/sources/pto-isa/include python run_tpushpop_cv.py
```
