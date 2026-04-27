# PTOAS feature requests (PTO MLIR dialect + Python bindings)

This document collects **actionable requests** for the PTOAS / PTO dialect stack so that **flash-attention–style kernels** written in Python (e.g. `examples/aot/flash_attention/experimental/fa_builder.py` via `ptodsl`) can **closely match** the hand-tuned reference in `examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp` (`runTFA`, `compute_qk`, `compute_p`, `compute_pv`, `compute_gu`).

Upstream sources referenced below live under:

`.agent/skills/translate_cpp2py/references/external_repo/PTOAS/`

---

## 1. Allow cube-side `pto.tpush` from non-ACC tiles (C2V producer coverage)

**Problem.** `TPushOp::getPipe()` only maps **`AddressSpace::ACC`** to a concrete pipe (`PIPE_FIX`). **`MAT`** and **`LEFT`** tiles map to **`PIPE_UNASSIGNED`**, so MLIR verification fails with *“tile type must map to a supported producer pipe”* when attempting to push an fp16 staging tile (e.g. post-`TCvt` from acc) over a C2V pipe.

**Evidence.** `include/PTO/IR/PTOOps.td`, `TPushOp` `getPipe()` (lines ~1767–1792): only `ACC` and `VEC` branches; all other address spaces return `PIPE_UNASSIGNED`.

**Motivation (ref FA).** The C++ reference keeps **fp32 QK in GM** and uses **fp16** inside vec macros for softmax output / P staging. A Python port naturally wants **cvt(acc f32 → mat/left f16) → tpush** to **halve `slot_size`** and vec FIFO pressure while keeping matmul in fp32.

**Ask.**

- Extend **`TPushOp`** (and verifier / lowering to EmitC) so **cube producers** can legally push **`TileBufType` in `MAT` and/or `LEFT`** with dtypes compatible with the pipe’s `slot_size`, **or**
- Document and implement an **official lowering path** (e.g. implicit MTE move acc→staging then push) so frontends do not need to guess unsupported combinations.

---

## 2. Decouple `slot_size` (wire bytes) from producer/consumer tile element type

**Problem.** `initialize_l2g2l_pipe` takes a single **`slot_size` (bytes)** while `tpush`/`tpop` tile types carry **dtype + shape**. Today authors must keep **manual consistency** between `SLOT_SIZE_QK`, cube `TileAcc<f32>`, and vec `Tile<Vec,f32,…>`; there is no first-class “**fp32 compute, fp16 wire**” contract.

**Evidence.** `InitializeL2G2LPipeOp` in `PTOOps.td` (~1681–1712): `slot_size` is a plain `i32`; pipe init does not encode logical vs physical width.

**Motivation (ref FA).** Reference layout uses **`sizeof(float)` × Cube_S0 × Tile_S1`** in GM for `qk_tile_fifo`, while vec tiles are **`Vec_S0 × Tile_S1`** with **`Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor`**. The toolchain should help express **logical tile**, **wire format**, and **vec working tile** without ad-hoc byte math in Python.

**Ask.**

- Optional attributes on **`initialize_l2g2l_pipe`** (or companion op) for **`wire_elem_type`**, **`logical_shape`**, and/or **`vec_slice_shape`**, validated against `slot_size`, **or**
- A small **tablegen-verified** bundle type for “pipe slot descriptor” consumed by both cube and vec builders.

---

## 3. First-class **K-split** (`kTileFactor`) and **partial QK** delivery to vec

**Problem.** The reference runs **`kTileFactor = Tile_S1 / Cube_S1`** cube passes (e.g. two **128×128** matmuls per **256**-wide logical tile), stores **`Cube_S0 × Cube_S1`** slices into GM, and vec **`compute_p`** performs **`kTileFactor`** **TLOAD**s of **`Vec_S0 × Cube_S1`** into a **`Vec_S0 × Tile_S1`** vec tile. The Python + `l2g2l_pipe` path instead tends toward **one full `Cube_S0 × Tile_S1` tpush** and a **`S0_HALF × S1_TILE` tpop**, which inflates **vec UB** versus **`Vec_S0 × Tile_S1`**.

**Motivation (ref FA).** Matching **`CUBE_S1`**, **`kTileFactor`**, and **`Vec_S0`** is required for both **numerics/scheduling parity** and **UB parity** with `fa_kernel.cpp`.

**Ask.**

- Either **documented** lowering from “ref-style GM layout + sync” to **`initialize_l2g2l_pipe` + `tpush`/`tpop`**, **or** new ops / pipe modes for:
  - **multiple ordered `tpush`es** per logical `tile_id` with **fixed GM packing** matching the reference’s `base_elems` formulas, and
  - **vec-side assembly** (`tpop` into column sub-ranges of one vec tile, or explicit `tassign`/`subview` at UB addresses) without requiring a single oversized **`tpop`** result tile.

---

## 4. Richer **`split`** / subblock model (beyond one `TILE_UP_DOWN` halving)

**Problem.** `split` on `tpush`/`tpop` models a **single** split axis enum; reference logic combines **`get_subblockid()`**, **`row_slice`**, and **`kTileFactor`** to address **four** distinct **32-row** bands across **`Cube_S0 = 128`**. Expressing that with only **one** up/down split per op forces **larger per-core vec tiles** than the reference.

**Evidence.** Design notes in `docs/designs/ptoas-tpush-tpop-design.md` (split semantics); reference `compute_p` row/col slicing in `fa_kernel.cpp`.

**Ask.**

- Consider **documented composition** of splits (e.g. nested phases) **or** additional split modes / **multi-phase tpop** that align with **`row_slice × subblock`** patterns used in FA macros.

---

## 5. **`local_slot_num` / vec `reserve_buffer`** vs GM-only consumer patterns

**Problem.** `local_slot_num` must be **> 0** and `local_addr` is mandatory for `initialize_l2g2l_pipe` (verifier in `PTO.cpp` / design doc §5.2). The reference often behaves like **“cube writes GM; vec reads GM after sync”** with **smaller vec-local FIFOs** (`srcVecTNBuffers`, etc.), not necessarily a full **local mirror** of every slot byte in UB.

**Evidence.** `docs/designs/ptoas-tpush-tpop-design.md` (~318–361, ~759–761).

**Ask.**

- Optional **GM-primary consumer** mode: vec **`tpop`** semantics that **do not** require **`reserve_buffer(slot_size × local_slot_num)`** when the consumer only needs a **bounded scratch** (with **verified** max live bytes), **or**
- A **`tpop_from_gm` / `wait_slot` + `load`** pattern with **verified** cross-core ordering equivalent to **`TSync_Custom`** in the reference.

---

## 6. Explicit **sync / event** surface in the dialect (parity with `TSync_Custom` / CV FIFO)

**Problem.** Reference FA uses **`TSync_Custom`**, **`should_wait_consumption` / `should_notify_consumption`**, and optional **CV comm** for backpressure. Python builders today lean on **`--enable-insert-sync`** and pipe **`tfree`** ordering; there is no close 1:1 mapping to **named sync tokens** and **FIFO depth** parameters from `fa_kernel.cpp`.

**Motivation (ref FA).** Tuning **`QK_PRELOAD`**, **`qkp_tile_fifo_size`**, and **`CV_FIFO_CONS_SYNC_PERIOD`** is central to the C++ launch.

**Ask.**

- Expose **optional** `record_event` / `wait_event` (or reuse existing async session ops if applicable) with **stable lowering** to the same primitives reference kernels use, **and/or**
- A **small FA template** in docs that maps **`runTFA` template parameters** → PTO ops + attrs.

---

## 7. Python bindings: **ergonomics** beyond raw `mlir` ODS

**Problem.** `python/pto/dialects/pto.py` is largely **generated ODS exports**; authors of large kernels still hand-roll **byte offsets**, **`slot_size`**, and **layout** in application code (`ptodsl` or otherwise), which is error-prone when **`S0`**, **`S1_TILE`**, or **`HEAD`** change.

**Ask.**

- **Optional** Python helpers (same package or `ptodsl`-side) for:
  - **Pipe bundle construction** (`dir_mask`, `slot_size`, `slot_num`, `local_slot_num`) with **static consistency checks**,
  - **UB layout** from a declarative map of **tile names → (space, dtype, shape)** with **overlap detection**,
  - **“Reference FA preset”** constants: `CUBE_S0`, `CUBE_S1`, `TILE_S1`, `QK_PRELOAD`, FIFO depths — emitting the right **`initialize_l2g2l_pipe`** / legacy `*_initialize_pipe` combo.

---

## 8. Documentation: **reference kernel ↔ PTO pipe** mapping

**Ask.** Add a short chapter to `docs/designs/ptoas-tpush-tpop-design.md` (or a new doc under `docs/designs/`) that shows:

1. How **`TSTORE(qkGlobalTile, qkAccTile)`** + **`TLOAD(qkVecSub, qkGlobalSub)`** in `fa_kernel.cpp` maps to **`initialize_l2g2l_pipe` + `tpush` + `tpop`** (including **GM stride** / **`kTileFactor`**).
2. Which **`split`** values approximate **`TileSplitAxis::TILE_UP_DOWN`** in the reference P headers.
3. **Known limitations** (e.g. **`TPushOp` producer address spaces** as of current `PTOOps.td`).

---

## Priority (suggested for FA parity)

| Priority | Item | Why |
|----------|------|-----|
| P0 | **1** (non-ACC `tpush`) + **2** (slot/dtype decouple) | Unblocks **fp16-on-wire** and smaller vec FIFO without losing fp32 matmul. |
| P0 | **3** (K-split / partial QK) | Matches **reference cube + vec geometry**; largest structural mismatch today. |
| P1 | **5** (GM-primary / smaller local ring) | Unlocks **`QK_PRELOAD = 4`**-class schedules without linear growth of vec **`reserve_buffer`**. |
| P1 | **6** (explicit sync) | Needed for **faithful** backpressure / CV parity when scaling blocks/cores. |
| P2 | **4** (richer split) | Reduces pressure on Python to fake **row_slice** with full-height tiles. |
| P2 | **7–8** (bindings + docs) | Reduces integration risk and documents the **intended** lowering contract. |

---

## Related artifacts in this repo

- Experimental FA builder: `examples/aot/flash_attention/experimental/fa_builder.py`
- Reference C++ kernel: `examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp`
- Broader gap narrative: `examples/aot/flash_attention/known_gap.md`
