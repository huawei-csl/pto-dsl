# split_pipe — archived PTO codegen (best attempt vs reference C++)

This folder holds a **frozen snapshot** of the closest match produced today between:

1. **PTO-DSL builder output** (`fa.pto.mlir`) — MLIR emitted by `kernels/fa_performance_builder.py`.
2. **ptoas-generated Ascend C++** (`fa.ptoas.generated.cpp`) — lowered from that MLIR with **`ptoas --pto-arch=a3 --pto-level=level3 --enable-insert-sync`**.
3. **Hand-written reference kernel** (`fa_performance_kernel.reference.cpp`) — copy of `../cpp_ref/split_pipe/kernels/flash_atten/fa_performance_kernel.cpp` for side‑by‑side comparison.

“Best attempt” here means the **default split_pipe product**: **`FA_NUM_TILES=16`**, **`FA_S1_TILE=512`**, **`FA_Q_ROWS=3072`** — i.e. the **`fa.so`** / **`fa.mlir`** path aligned with the bundled **`compile.sh`** defaults and the cpp_ref benchmark geometry documented in `../README.md` (**HEAD=128**, wide softmax tile).

---

## How these files were generated

From `split_pipe/`:

```bash
# Rebuild only the plain fa.so variant (NUM_TILES=16 → plain filenames).
FA_TILES=16 FA_S1_TILE=512 FA_Q_ROWS=3072 bash compile.sh
```

Pipeline (same as `compile.sh`):

| Step | Command | Output |
|------|---------|--------|
| 1 | `FA_NUM_TILES=16 FA_S1_TILE=512 FA_Q_ROWS=3072 python kernels/fa_performance_builder.py` | stdout redirected to `build_artifacts/fa.mlir` |
| 2 | `ptoas --pto-arch=a3 --pto-level=level3 --enable-insert-sync build_artifacts/fa.mlir` | stdout redirected to `build_artifacts/fa.cpp` |
| 3 | `bisheng … caller.cpp -o build_artifacts/fa.so` | linked host shim + embedded kernel C++ (not duplicated here) |

Files in **this directory** are copies of steps **1–2** plus the reference source (run from **`split_pipe/`**, i.e. the parent of `codegen_results/`):

```bash
cp build_artifacts/fa.mlir               codegen_results/fa.pto.mlir
cp build_artifacts/fa.cpp                codegen_results/fa.ptoas.generated.cpp
cp ../cpp_ref/split_pipe/kernels/flash_atten/fa_performance_kernel.cpp \
   codegen_results/fa_performance_kernel.reference.cpp
```

Canonical reference path from **`split_pipe/codegen_results/`** alone:

`../../cpp_ref/split_pipe/kernels/flash_atten/fa_performance_kernel.cpp`

**Naming:** The DSL emits **MLIR** (`module { … func.func @cube_kernel … }`). There is no separate `.pto` file extension in this repo; **`fa.pto.mlir`** is the **PTO-DSL MLIR IR** artifact users diff against `ptoas`.

---

## Gaps: `fa.ptoas.generated.cpp` vs `fa_performance_kernel.reference.cpp`

Both ultimately include **`pto/pto-inst.hpp`** and lower to **`TPipe` / `TLOAD` / `TMATMUL` / vec ops**, but structure and intent differ substantially.

### 1. Kernel shape and entry points

| Aspect | Reference | DSL → ptoas |
|--------|-----------|-------------|
| Top-level | Template **`runTFA<…>`** fuses QK / softmax (**`compute_p`**) / PV / GU (**`compute_gu`**) in one TU | **`cube_kernel`** (QK + PV matmuls and FIFO I/O) and **`vector_kernel`** (softmax + GU side) as **separate `AICORE` functions**, plus **`call_both`** wrapper |
| Launch API | **`LaunchTFA<…>`** with many GM FIFO / profiling pointers (`fa_performance_kernel.h`) | **`call_both`** (`caller.cpp`) passes **`gm_slot`** scratch + Q/K/V/O tensors |

The reference is optimized around **macro helpers** (`pto_macro_matmul.hpp`, `pto_macro_fa_softmax.hpp`, `pto_macro_fa_gu.hpp`). The DSL path expresses the same *logical* pipeline via **`pto.*` MLIR ops**, which **ptoas** expands into long SSA-style **`Tile`/`GlobalTensor`** code without those FA macros.

### 2. Compile-time geometry

Reference defaults (see `fa_performance_kernel.h`) include **`kFaQkPreload = 4`**, **`kFaTileS1 = 256`**, **`kFaCubeS1 = 128`**, while **`LaunchTFA`** is instantiated from cpp_ref `run.py`/JIT with parameters that may override **`TILE_S1`**.

This archived DSL build fixes **`S0 = 32`** (vec half-rows **`16`** per sub-block), **`S1_TILE = 512`**, **`HEAD = 128`**, **`NUM_TILES = 16`**, **`QK_PRELOAD = 2`** in `fa_performance_builder.py`. So numerically “closest” is about **matching workload size** (e.g. **8192** sequence length), **not** identical template parameters to every **`kFa*`** constant in the header.

### 3. FFTS pipes and flag namespaces

Reference constructs **`TPipe<BUF0_QK_READY, …>`**, **`TPipe<BUF1_SM_READY, …>`**, **`TPipe<UPDATE_READY, …>`** using **`FftsBufferFlag`** (`BUF0_QK_READY`, …).

ptoas emits **`TPipe<0, Direction::DIR_C2V, …>`**, **`TPipe<2, …>`**, **`TPipe<4, Direction::DIR_V2C, …>`** (see generated file near **`cube_kernel`**). The **numeric first template arguments (0 / 2 / 4)** line up with **FFT slot ordering** chosen by the DSL **`initialize_l2g2l_pipe` / `aic_initialize_pipe`** sequence — **not** identical to the reference’s **QK → P → PV** `TPipe` declaration order (DSL initializes **QK → PV → P**).

### 4. Synchronization and tail behavior

Reference **`runTFA`** ends with explicit **CV / FFTS** coordination (**`wait_flag_dev(CV_BLOCK_END)`**, **`ffts_cross_core_sync`**, etc., conditioned on **`DAV_CUBE`/`DAV_VEC`**).

Generated code finishes vector/cube sections with **`ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll)`**, which lowers to **`pipe_barrier(PIPE_ALL)`** (see top of `fa.ptoas.generated.cpp`). That is the **`ptoas --enable-insert-sync`** story — simpler and **not** a line‑by‑line match to the reference tail.

### 5. Dependencies and portability

Reference pulls **`acl`**, **`Pto_prefetch`**, **`TSyncCVID`**, FA‑specific macro headers.

Generated kernel code is intentionally minimal (**`pto-inst.hpp` + auto-sync helper**). Host linkage still uses **`caller.cpp`** / **`set_ffts_base_addr`** in the full build.

---

## Using this archive

- Diff MLIR iterations: `diff -u fa.pto.mlir …`
- Diff lowered C++ vs reference: `diff -u fa.ptoas.generated.cpp fa_performance_kernel.reference.cpp` (expect **large** differences — use the table above to interpret).
- After changing **`fa_performance_builder.py`**, regenerate **both** `build_artifacts/` and **these snapshots** so documentation stays reproducible.
