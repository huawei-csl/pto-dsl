# `ir_ref/fa.cpp` — compile, launch, correctness / perf

This folder mirrors the **`split_pipe`** flow (`caller.cpp` + `bisheng` shared library + `torch_npu` Python runner), but compiles the **checked-in IR reference** [`../fa.cpp`](../fa.cpp) produced from [`../fa.pto`](../fa.pto) (`ptoas --pto-arch=a3 --pto-level=level3 --enable-insert-sync`).

## Baked-in geometry

| Symbol | Value |
| ----- | ----- |
| `Q_ROWS` | 2048 |
| `HEAD` | 128 |
| `S1_TOTAL` | 4096 |
| `S1_TILE` | 256 (`NUM_TILES` = 16) |

Host tensors and GM scratch sizes come from `split_pipe/kernels/fa_performance_builder.py` with **`FA_Q_ROWS=2048`**, **`FA_S1_TILE=256`**, **`FA_NUM_TILES=16`** (reload via `fa.build_env` after compile).

**Launch grid:** this kernel divides **`NUM_TILES` (16)** across `get_block_num()`, not `NUM_Q_BLOCKS`. The runner therefore uses **`blockDim = min(NUM_TILES, num_cube_cores)`** (same idea as mapping tile loops to the grid). Using `NUM_Q_BLOCKS` here triggers device faults.

## Setup

From the `pto-dsl` repo root (Python imports `ptodsl`, `torch_npu`):

```bash
cd /workdir/pto-dsl
pip install -e .
```

Environment (same as other `examples/aot` demos):

- `ASCEND_TOOLKIT_HOME`
- `PTO_LIB_PATH` — directory containing `include/pto/` (default in many images: `/sources/pto-isa`)

## Build `fa.so`

```bash
cd /workdir/pto-dsl/examples/aot/flash_attention/ir_ref/launch_kernel
bash compile.sh
```

Produces `build_artifacts/fa.so` and `build_artifacts/fa.build_env`.

Regenerate `../fa.cpp` from PTO when needed:

```bash
cd /workdir/pto-dsl/examples/aot/flash_attention/ir_ref
bash gen_cpp.sh   # or: ptoas --pto-arch=a3 --pto-level=level3 --enable-insert-sync fa.pto > fa.cpp
```

## Run correctness + benchmark

```bash
cd /workdir/pto-dsl/examples/aot/flash_attention/ir_ref/launch_kernel
FA_BENCH_NO_PLOT=1 python3 run.py
```

Optional: `FA_BENCH_LENGTHS=4096` (default is already `4096`).

## Results on reference hardware (reproduced here)

**Environment:** CANN 8.5.0 toolkits, `bisheng --npu-arch=dav-2201`, `ptoas` from `/installers/ptoas-cli/bin/ptoas`, **dav‑2201** NPU, **24** cube cores (see `get_num_cube_cores()`), date **2026-04-30**.

| Step | Outcome |
| ---- | ------- |
| **`bash compile.sh`** | **Success** (~4s); links `../fa.cpp` via `-DKERNEL_CPP=...` |
| **`python3 run.py`** | **Fails** at `torch.npu.synchronize()` after `call_kernel`: **ACL 507015**, **aicore exception**, **CCU instruction address check** (vector core backtrace in device log) |

So **`fa.cpp` does not reach a completed device execution** in this configuration; **kernel latency / TFLOP/s are not reported** for it.

**Related checks on the same machine:**

- Applying the **`wait_flag_dev` / `ffts_cross_core_sync`** CV handshake from [`split_pipe/debug_cpp/forward_debug/README.md`](../../split_pipe/debug_cpp/forward_debug/README.md) to a copy of `fa.cpp` **still hit 507015** here — the IR layout differs from the forward-debug S256 kernel that clears the sync fault.
- **`split_pipe`** AOT at **`FA_S1_TILE=512`**:** same-style 507015** on synchronize (see [`split_pipe/README.md`](../../split_pipe/README.md) §A7).
- **`split_pipe`** AOT at **`FA_S1_TILE=256`**:** synchronize completes** but **`torch.testing.assert_close` fails** (NaNs / large drift — §A11).
- **`split_pipe/debug_cpp/forward_debug`** (`fa.ptoas.forward_edited.cpp`): **synchronize completes**; **numerics still fail** vs fp32 reference (documented there).
- **Baseline hand-written JIT** [`cpp_ref/split_pipe/run.py`](../../cpp_ref/split_pipe/run.py) **passes** correctness on this NPU. Closest reported row (**same `S1=4096`, `HEAD=128`, but `Q_ROWS=3072` and `tile_s1=512`**, not the IR reference geometry):

```bash
cd /workdir/pto-dsl/examples/aot/flash_attention/cpp_ref/split_pipe
python3 run.py
# excerpt (S1=4096 case), 2026-04-30:
#   JIT flash kernel           : 0.120 ms/iter  (54.307 TFLOP/s)
#   npu_fused_infer_attention  : 0.254 ms/iter  (25.628 TFLOP/s)
```

Use that run as an upper-bound sanity reference only; it is **not** the same kernel or tensor shapes as `ir_ref/fa.cpp`.

## Files

| File | Role |
| ---- | ---- |
| `compile.sh` | `bisheng` → `build_artifacts/fa.so` |
| `caller.cpp` | `call_kernel` → `call_both<<<blockDim, …>>>` (same pattern as `split_pipe/caller.cpp`) |
| `run.py` | Loads `.so`, correctness vs `fa_reference`, bench vs `torch_npu.npu_fused_infer_attention_score` |
