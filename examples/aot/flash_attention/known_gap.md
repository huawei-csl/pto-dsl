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
| [`pto_macro_fa_softmax.hpp`](../../../../pto-isa-master/kernels/manual/common/flash_atten/pto_macro_fa_softmax.hpp) | Streaming softmax: `softmax_opt_fa_init_impl` / `softmax_opt_fa_not_init_impl`, scale = `1/sqrt(HEAD)`, **TROWMAX**, **TROWEXPANDSUB**, **TEXP**, **TROWSUM**, **reshape + TCVT** to fp16, causal branches where applicable. |
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

| `S0=128` (Apr 2026) | Default `S0` raised to **128** (`FA_S0`). `pto.tpush` from cube still requires **ACC** tiles (PTO verifier: only `AddressSpace::ACC` maps to a producer pipe); staging QK as **fp16 on MAT/LEFT** before push was rejected at MLIR verify, so QK stays **fp32 on the wire** with full `SLOT_SIZE_QK`. Vec softmax reuses **`p_fp32` as `row_max` scratch** (same lifetime as before `row_expand_sub`) plus a **single shared `VEC_RECV_OFF`** sized for the larger of QK/PV half-tiles. `experimental/run.py` + `compile.sh` pass on NPU at ~24 TFLOP/s (unchanged order-of-magnitude vs fused ref). |

---

## PTOAS / PTO dialect / Python binding — feature requests (algorithm parity)

These are the main **toolchain** gaps noticed while aligning `experimental/fa_kernel` with `cpp_ref/naive_tpush/fa_kernel.cpp`. They are not criticisms of the hand-written reference; they are concrete asks so the **same algorithm config** (tiling, dtypes on wires, vec working set) can be expressed without fighting verifiers or UB.

1. **C2V `pto.tpush` producer tiles beyond ACC**  
   Today `TPushOp::getPipe()` maps **only** `AddressSpace::ACC` → `PIPE_FIX` (see `PTOOps.td`); **MAT** and **LEFT** producers yield `PIPE_UNASSIGNED` and fail verification. The reference keeps QK in **fp32 in GM** (`qk_tile_fifo`) and uses **fp16** only inside vec macros (`TileDataH_T`, `TCVT`). A natural DSL port would **cvt** `TileAcc<f32>` → **`Tile<Mat|Left,f16>`** and `tpush` that tile to halve **`slot_size`** / vec FIFO pressure. **Ask:** allow **fp16 (and/or LEFT/MAT) tiles** as legal C2V `tpush` sources when `slot_size` matches, or document the intended lowering (e.g. MTE path) so Python does not need ACC-only staging.

2. **Decouple `slot_size` from “one full cube row tile” for vec UB accounting**  
   Reference **`Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor`** (e.g. **32** rows × **256** cols in vec UB) while GM still holds **`Cube_S0 × Tile_S1`** floats per logical tile, assembled from **`kTileFactor`** slices of **`Cube_S0 × Cube_S1`**. The DSL **`l2g2l_pipe`** ties **vec `reserve_buffer`** size to **`SLOT_SIZE_QK`** and **`tpop`** delivers **`S0_HALF × S1_TILE`** per subblock. **Ask:** first-class **“logical tile vs wire chunk”** (multi-slot per tile_id, or column-strip `tpop` into a fixed vec workspace) so vec UB tracks **`Vec_S0`** like the C++ launch, not **`Cube_S0/2`** per `TILE_UP_DOWN` alone.

3. **`kTileFactor` / K-split + softmax without a single 64×256 vec tile**  
   Matching the reference requires **multiple `compute_p` / `row_slice` passes** per tile and **partial QK layout in GM** (`base_elems + row_offset * Cube_S1`). **Ask:** DSL helpers or ops for **GM strided views** + **event sync** equivalent to `TSync_Custom` / `qk2smSync`, or **documented** mapping from `initialize_l2g2l_pipe` + `tpop` to that pattern so cube can emit **128×128** stores while vec runs **32×256** softmax without holding a **64×256** `qk_vec` buffer per subblock.

4. **`QK_PRELOAD = 4` and deeper CV FIFOs**  
   Reference uses **`qkPreloadNum = 4`** with **`l1_exp_max_ififo[qkp_tile_fifo_size]`**. DSL stays at **`QK_PRELOAD = 2`** for a smaller **`exp_max` ring**. **Ask:** either **lowered UB cost** for pipe rings (item 1–2) or **optional GM-backed vec inputs** so preload depth can match the C++ launch without manual byte arithmetic.

5. **Python binding ergonomics**  
   **Ask:** optional **computed layout** (or static asserts) from tensor shapes for **MAT / VEC base offsets** so raising `S0` cannot silently overlap **`MAT_P_FIFO`** with cube tiles; and a **single knob** mirroring `runTFA` template parameters (`CUBE_S0`, `CUBE_S1`, `TILE_S1`, `QK_PRELOAD`, CV FIFO depth) mapped to **`S0`**, **`S1_TILE`**, **`QK_PRELOAD`**, and pipe **`slot_num` / `local_slot_num`**.

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
- [ ] **Softmax:** Port **`softmax_opt_fa_init_impl`** and **`softmax_opt_fa_not_init_impl`** (and causal paths if needed) as an explicit sequence of DSL ops matching **TROWMAX → TROWEXPANDSUB → scale → TEXP → TROWSUM → reshape/TCVT** behavior in `pto_macro_fa_softmax.hpp`.
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
