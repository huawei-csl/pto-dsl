# Flash Attention — split_pipe (PTO-DSL builder)

Python builder `kernels/fa_performance_builder.py` emits MLIR → `ptoas` → C++ → `bisheng` `.so`, following the same **software-pipelined** FA schedule as the reference hand-written kernel in `../cpp_ref/split_pipe/kernels/flash_atten/fa_performance_kernel.cpp` (QK preload, softmax/GU overlap, `exp_max` ping-pong). The DSL uses explicit pipe helpers (`initialize_l2g2l_pipe`, `tpush`/`tpop` with `TILE_UP_DOWN`) instead of `TMPipe` + `TileSplitAxis` from the C++ file.

## Current status (vs bundled C++ reference)

| Check | Result |
| ----- | ------ |
| **`../cpp_ref/split_pipe/run.py`** (JIT `fa_performance_kernel.cpp`) | Passes correctness + benchmark on NPU in this environment |
| **`python run.py`** here — **default** (`S1_TILE=512`, `Q_ROWS=3072`, default `FA_NUM_TILES` matching `fa.so`) | **Fails** at `torch.npu.synchronize()` with **ACL 507015** / **CCU instruction address check** (aicore fault); numerics not validated |
| **`FA_S1_TILE=256`** + matching compile/runtime env | Kernel often **finishes** without that sync fault but **`torch.testing.assert_close` fails** (NaNs / large errors) — **not yet a validated port** |

So the Python port does **not** yet **run correctly** nor **fully match** the C++ reference at the default cpp_ref-shaped geometry (`HEAD=128`, `tile_s1=512`, large Q).

### Likely remaining gaps (see `DSL_FIX_TODOS.md`)

1. **Compile/runtime env parity:** `FA_NUM_TILES`, `FA_S1_TILE`, and `FA_Q_ROWS` are fixed in the emitted MLIR. **`compile.sh`** writes **`build_artifacts/fa${TAG}.build_env`** per variant; **`run.py`** picks the file whose **`FA_NUM_TILES * FA_S1_TILE`** equals the first **`FA_BENCH_LENGTHS`** entry (tie-break: **`FA_S1_TILE`** env if set, else **`fa.build_env`**), then **`importlib.reload`**s **`fa_performance_builder`** so Python GM/tensor shapes match the loaded `.so`. You can still override via exported **`FA_*`** before launch if needed.
2. **FFTS / CV plumbing:** Hand-written `runTFA` constructs **`TPipe`** objects in **QK → P (V2C) → PV** order (`BUF0_QK_READY`, `BUF1_SM_READY`, `UPDATE_READY`). The DSL builder calls **`initialize_l2g2l_pipe`** for **QK** and **PV** first, then **`aic_initialize_pipe` / `aiv_initialize_pipe`** for **P** (**QK → PV → P**). Generated **`TPipe<first_template_arg, …>`** indices may therefore differ from the reference header ordering; **`ptoas --enable-insert-sync`** supplies multipipe synchronization. **Kernel-tail** **`wait_flag_dev` / `ffts_cross_core_sync`** vs **`ptoas_auto_sync_tail`** remains an **A8** structural gap (see **`DSL_FIX_TODOS.md`**).
3. **`S1_TILE=512` vec path:** Errors correlate with the widest softmax tiles (vs `examples/aot/flash_attention/experimental/` at `S1_TILE=256`, which passes).

### Debug workflow (recommended)

1. Regenerate and diff **`build_artifacts/fa.cpp`** against **`../cpp_ref/split_pipe/kernels/flash_atten/fa_performance_kernel.cpp`** for pipe IDs (`TPipe<first_template_arg, …>`), schedule phases, and sync tails.
2. After any builder change, **`bash compile.sh`** with the **same** `FA_*` env vars you will use for **`python run.py`**.
3. Compare behaviour with **`../cpp_ref/split_pipe`** on the **same** NPU.

---

## Shapes & defaults

**`HEAD=128`** and default **`S1_TILE=512`** match `../cpp_ref/split_pipe/run.py` (`test_flash(..., head=128)`, typical `tile_s1=512`). Cube rows per block stay **`S0=32`** (vector softmax tiles **`[S0_HALF, S1_TILE]`**); total Q rows default to **`3072` (`128 * 24`)** like the cpp_ref benchmark `s0`. Override with **`FA_Q_ROWS`** (must match at compile and run).

Optional **`FA_S1_TILE`** (default `512`) is for experiments; changing it changes **`NUM_TILES`** needed for the same sequence length (`seq_len = NUM_TILES * S1_TILE`) and **must** be rebuilt.

The reference C++ kernel uses larger cube blocks (`CUBE_S0=128`) with narrower logical vec rows; this DSL example keeps the geometry that fits explicit UB layouts (`S0=32`).

---

## Setup

```bash
cd /workdir/pto-dsl
pip install -e .
```

Environment (same as other `examples/aot` demos):

- `ASCEND_TOOLKIT_HOME`
- `PTO_LIB_PATH` — directory that contains the `include/` tree with `<pto/pto-inst.hpp>` (repo root or `.../include`)

---

## Build kernels

From this directory:

```bash
bash compile.sh
# Optional: FA_TILES=16,64 bash compile.sh
```

`compile.sh` passes through:

- **`FA_NUM_TILES`** — per variant (from `FA_TILES`).
- **`FA_S1_TILE`** — default `512`.
- **`FA_Q_ROWS`** — default `3072`.

Produces `build_artifacts/fa.so`, `fa_32.so`, … (each variant corresponds to `FA_NUM_TILES` baked into the emitted IR), plus matching **`fa.build_env`**, **`fa_32.build_env`**, … (`FA_NUM_TILES`, `FA_S1_TILE`, `FA_Q_ROWS`) for **`run.py`** auto-sync.

**Example — 8k sequence with default tile width:** `NUM_TILES=16`, `S1_TILE=512` → `fa.so` + `fa.build_env`.

---

## Correctness + benchmark

```bash
python run.py
```

Override sequence lengths: `FA_BENCH_LENGTHS=8192,32768 python run.py`.

**Important:** after **`compile.sh`**, **`run.py`** reads **`fa*.build_env`** for the first benchmark length so imported builder constants stay aligned with how that `.so` was built. If you rebuild with different **`FA_S1_TILE`** / **`FA_Q_ROWS`**, re-run **`compile.sh`** (or set matching **`FA_*`** env vars yourself). Ambiguous lengths (e.g. `8192 = 16×512 = 32×256`) require **`FA_S1_TILE`** in the environment or only one matching **`*.build_env`** on disk.

---

## Compare generated C++ to reference

Diff `build_artifacts/fa.cpp` (or `fa_<N>.cpp`) against `../cpp_ref/split_pipe/kernels/flash_atten/fa_performance_kernel.cpp`: scheduling intent should align on pipeline phases; surface syntax differs (`pto.call` cube/vec vs fused `runTFA`, etc.).
