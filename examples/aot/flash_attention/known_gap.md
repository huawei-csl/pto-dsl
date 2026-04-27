# Known gap: PTO Python DSL flash attention vs reference C++

This document compares the AOT flash-attention builders (`fa_builder.py`, `experimental/fa_builder.py`) and their `ptoas` output to the hand-written reference in `cpp_ref/naive_tpush/fa_kernel.cpp` (`runTFA` and helpers). It updates earlier notes on **macro parity** and **`--enable-insert-sync`**.

## Revised summary

### What the DSL already mirrors

- High-level **software pipeline**: QK preload, steady-state interleaving of softmax (lookahead) with GU (current tile), and an **`exp_max` ping-pong ring** when `QK_PRELOAD == 2` (experimental) to match the reference’s “softmax ahead of GU” hazard story.
- **Multi-pipe** QK (cube→vec), P (vec→cube), PV (cube→vec) with GM-backed slots, analogous to the reference’s FIFO staging (different mechanism, same role).

### Primary performance gaps (largest expected impact)

1. **Cube tiling and S1 sub-tiling**  
   Reference: `CUBE_S0 = 128`, `CUBE_S1 = 128`, `TILE_S1 = 256`, so **`kTileFactor = 2`** (two 128-wide K slices per logical 256-wide tile).  
   DSL (experimental): **`S0 = 128`** by default (env `FA_S0`), still a **single** **`S1_TILE = 256`** matmul per tile (no K-split). The row-block size now matches reference **M**; the remaining gap is **K micro-tiling / matmul overlap** versus the reference’s two `Cube_S1` passes per logical tile.

2. **Real L1 double-buffering**  
   Reference: `kMatTNBuffers = 2`, `pMatTNBuffers = 2`, `vMatTNBuffers = 2`, plus vec-side ping-pong (`srcVecTNBuffers`, `xexpVecTNBuffers`, `outOTileNBuffers`).  
   DSL: cube **single-buffered** tiles with **aliased** `[k_mat_s, k_mat_s]` / same for P/V; **`QK_LOCAL_SLOT_NUM = 1`** on the QK pipe because deeper local slots overflow vec UB. That limits overlap of **TLOAD** with **TMATMUL** compared to the reference.

3. **Preload depth**  
   Reference launch uses **`QK_PRELOAD = 4`** (`fa_kernel.cpp`). Experimental DSL uses **`QK_PRELOAD = 2`**. Shallower preload reduces cube/vec overlap on long S1. A DSL port to 4 needs a **4-deep `exp_max` ring** and more VEC space; an attempt faulted until UB layout (recv scratch, `MAT_P_FIFO` tail) is redesigned.

### Macro parity (item 5) — port to Python DSL, not “optional tuning”

The reference’s hot path is not arbitrary `tile.*` soup; it goes through shared headers that should be **replicated in Python DSL** so lowering and scheduling stay aligned with the tuned C++ path:

| Reference include | Role |
|-------------------|------|
| [`pto_macro_matmul.hpp`](../../../../pto-isa-master/kernels/manual/common/flash_atten/pto_macro_matmul.hpp) | Cube matmul with **`AccMode`** (e.g. `InitFinalSum` under `UF_ENABLE`), K tiling, and L0-oriented constraints. |
| [`pto_macro_fa_softmax.hpp`](../../../../pto-isa-master/kernels/manual/common/flash_atten/pto_macro_fa_softmax.hpp) | Streaming softmax: `softmax_opt_fa_init_impl` / `softmax_opt_fa_not_init_impl`, scale = `1/sqrt(HEAD)`, **TROWMAX**, **TROWEXPANDSUB**, **TEXP**, **TROWSUM**, and (inside the macro) the sequence that feeds **P**—including **TCVT** where the **macro** emits half for the V2C pipe. **QK** in the reference kernel is still **fp32 in GM** via **`TSTORE`** and **`TLOAD`** in `compute_qk` / `compute_p`; do not conflate macro-internal P conversion with a separate invented “fp16 QK wire”. |
| [`pto_macro_fa_gu.hpp`](../../../../pto-isa-master/kernels/manual/common/flash_atten/pto_macro_fa_gu.hpp) | **pto_macro_fa_gu** (`TROWEXPANDMUL` + `TADD`), **pto_macro_fa_gu_last** (+ `TROWEXPANDDIV` by `new_global_sum`), **pto_macro_fa_gu_single_and_last_tile**. |

**Goal:** Express the same ordered primitive sequence (and the same init vs non-init / last-tile branching) in `ptodsl` `tile.*` / `pto.*` APIs—or add thin DSL helpers that document a 1:1 mapping to those macros—so the compiler stack can match the reference kernel’s numerics and fusion expectations. Today’s DSL uses composable `tile.row_max`, `tile.exp`, etc.; they must be **audited and aligned** macro-step by macro-step, not assumed equivalent.

### `ptoas --enable-insert-sync` (item 6)

**`--enable-insert-sync` is intentional:** it simplifies generated C++ by having the toolchain insert synchronization. Impact on performance is treated as **minor** relative to **tiling, real double-buffering, and preload depth**.

Closing the gap should **not** rely on turning sync insertion off; it should rely on **geometry + buffering + macro-faithful lowering** (and any future DSL-level sync refinement if needed).

### Secondary / structural differences

- **GM / FFTS / CV:** Reference uses FFTS base, `TSync_Custom`, optional CV comm for many blocks; DSL uses `l2g2l_pipe` / `aic_initialize_pipe` / `aiv_initialize_pipe` and GM slot buffers. Functionally similar staging; details may diverge under high block counts.
- **Benchmark shape parity:** e.g. `experimental/run.py` uses `Q_ROWS` from the builder vs `naive_tpush/run.py` using `s0 = 128*24`; compare throughput with **matched** `Q_ROWS`, `HEAD`, `S1`, and tile counts when isolating kernel quality.

---

## Progress log (experimental `fa_builder.py`, Apr 2026)

Measured on NPU via `experimental/run.py` (Q=2048, H=128, S1_TILE=256): kernel holds ~24–26 TFLOP/s vs ~60+ TFLOP/s for `torch_npu` fused ref on the same script; correctness (`assert_close` at `run.py:151`) remains the gate.

| Change attempted | Result |
|------------------|--------|
| `QK_PRELOAD=4` + four `exp_max` slots + quad-unrolled vec/cube + true dual MAT banks for K/P/V | AICore CCU address fault (`mte`/`ccu`); likely VEC `tpop` scratch / expanded red region + dual `RIGHT` typing; **reverted**. |
| True L1 ping-pong (separate `MAT_K0`/`MAT_K1`, …) without preload-4 | Overlapped `MAT_P_FIFO` with tail tiles until `MAT_P_FIFO_OFF` was recomputed; still faulted with dual `RIGHT` `alloc_tile` until fully reverted to aliased single-buffer layout. |
| K-split: two `CUBE_S1=128` matmuls per tile via `tile.subview` on `qk_acc` | Builds and passes `assert_close`; **~7% slower** than one `S1_TILE=256` matmul on this target — **reverted**. |
| Reorder steady cube step (PV before K load) | **Slight regression** vs original order on sampled runs — **reverted**. |

**Takeaway:** With **`S0=128`** landed in the experimental builder, the largest remaining structural gaps versus the reference are **`kTileFactor` / `CUBE_S1` K-split**, **preload / ring depth (`QK_PRELOAD`, CV FIFO)**, and **vec working-tile geometry (`Vec_S0`)** relative to what `l2g2l_pipe` + `TILE_UP_DOWN` imply for UB. Raising `QK_PRELOAD` still needs more `exp_max` slots and recv/GM budget.

| `S0=128` (Apr 2026) | Default `S0` raised to **128** (`FA_S0`). Cube **`pto.tpush`** uses **`ACC`** QK tiles into `l2g2l_pipe` (same as today’s supported producer); the **reference** path is **`TSTORE`** from acc to **fp32 GM**, not MAT/LEFT→GM—**`PIPE_UNASSIGNED` for MAT/LEFT `tpush` is expected**, not a toolchain defect to “fix” for FA. Builder keeps **fp32 `SLOT_SIZE_QK`**, vec **`p_fp32` as `row_max` scratch**, and a **shared `VEC_RECV_OFF`** for half-tile **`tpop`** scratch. `experimental/run.py` + `compile.sh` pass on NPU at ~24 TFLOP/s (order-of-magnitude below fused ref). |

---

## PTOAS / PTO dialect / Python binding — reasonable asks (see `ptoas_request.md`)

Feature requests must **mirror what `fa_kernel.cpp` already does**, not invent paths the reference does not use (e.g. **no** MAT/LEFT→GM for QK, **no** extra **`tile.cvt`** pipeline on QK beyond what the **macros** imply for **P**). FP32 QK in the ref is **`TSTORE`/`TLOAD`** to/from **fp32 GM**; **`ptoas --enable-insert-sync`** remains the **intentional** allowed divergence from hand-placed `TSync_Custom`.

A maintained list of **documentation-first** and **parity** asks (GM packing vs `l2g2l_pipe`, `kTileFactor`/`Vec_S0`, sync equivalence, ergonomics) lives in **`examples/aot/flash_attention/ptoas_request.md`**.

---

## TODO: close the gap (Python DSL ↔ reference C++)

Use this as a work backlog; order roughly reflects suggested priority (tiling/buffers first, then macro fidelity, then integration).

### Tiling and cube schedule

- [ ] **Match reference cube geometry:** `CUBE_S0=128`, `CUBE_S1=128`, `TILE_S1=256`, and **`kTileFactor`** loop (two K slices per 256-wide tile) in the DSL builder’s cube kernel, or justify an equivalent FLOP/memory contract with measurements. *(Prototype K-split only: numerics OK, throughput down on current NPU.)*
- [x] **Match reference `CUBE_S0` (128) in experimental builder** — default `S0=128` via `FA_S0` (Apr 2026); UB layout was tightened (shared `tpop` recv sizing, `row_max` scratch reuse, smaller `VEC_RED_STRIDE`). Smaller blocks remain available with `FA_S0=32` etc. if needed.
- [ ] **Align `QK_PRELOAD`** with the reference launch (**4**) and extend the **`exp_max` / GU ring** logic (or equivalent hazard avoidance) for that depth; assert fifo and UB sizing.

### Double-buffering and overlap

- [ ] **Implement true L1 ping-pong** for **K**, **P**, and **V** cube tiles (separate physical buffers, not aliased `[x, x]`).
- [ ] **Vec UB layout:** budget space for **`QK_LOCAL_SLOT_NUM > 1`** if required to mirror reference QK pipe depth, without exceeding UB limits; coordinate with `SLOT_NUM` / GM stride math.
- [ ] **Vec tile banks:** mirror **`srcVecTNBuffers=2`**, **`xexpVecTNBuffers=2`**, **`outOTileNBuffers=2`** (or document why a smaller depth is equivalent).

### Macro parity in Python (lowering contract)

- [ ] **Matmul:** Port or wrap **`pto_macro_matmul`** semantics in DSL—especially **`AccMode`** (`Init` / `InitFinalSum` / partial vs final slices) and the **K-subslice** interaction with double-buffered K tiles.
- [ ] **Softmax:** Port **`softmax_opt_fa_init_impl`** and **`softmax_opt_fa_not_init_impl`** (and causal paths if needed) as an explicit sequence of DSL ops matching the **macro** ordering in `pto_macro_fa_softmax.hpp` (including whatever the macro uses to produce **half** for **P**—e.g. **`TCVT`** inside the macro—not a separate invented QK cast).
- [ ] **GU:** Port **`pto_macro_fa_gu`**, **`pto_macro_fa_gu_last`**, and **`pto_macro_fa_gu_single_and_last_tile`** as DSL sequences matching **TROWEXPANDMUL / TADD / TROWEXPANDDIV** usage in `pto_macro_fa_gu.hpp`.
- [ ] **Numerics test:** Keep **`torch.testing.assert_close`** (same `rtol`/`atol` as `run.py` / `experimental/run.py`) as the gate after each macro block port.

### Toolchain and integration

- [ ] **Keep `--enable-insert-sync`** in `compile.sh` / `experimental/compile.sh`; optimize kernel structure first; only revisit sync policy if profiling shows it dominates after tiling/buffer parity.
- [ ] **Optional:** Parity for **FFTS / CV / `TSync_*`** paths if multi-block or multi-core scaling diverges from reference after cube/vec parity work.
- [ ] **Docs:** When a builder variant reaches parity, record fixed constants (`CUBE_S0`, `CUBE_S1`, `TILE_S1`, `QK_PRELOAD`, buffer counts) next to `jit_util_flash.py` / `fa_kernel.cpp` launch parameters so drift is obvious in review.

---

## File map

| Artifact | Path |
|----------|------|
| Reference kernel | `cpp_ref/naive_tpush/fa_kernel.cpp` |
| JIT constants | `cpp_ref/naive_tpush/jit_util_flash.py` |
| Non-experimental builder | `fa_builder.py` |
| Experimental builder | `experimental/fa_builder.py` |
| PTO FA macros (ISA tree) | `pto-isa-master/kernels/manual/common/flash_atten/pto_macro_*.hpp` |
