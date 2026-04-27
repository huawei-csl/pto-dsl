# PTOAS feature requests (PTO MLIR dialect + Python bindings)

This document lists **reasonable** asks for PTOAS / the PTO dialect so Python FA (`experimental/fa_builder.py` via `ptodsl`) can **mirror** `examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp` (`runTFA`, `compute_qk`, `compute_p`, `compute_pv`, `compute_gu`).

**Ground rules (read first)**

1. **No invented algorithms.** PTOAS should support a **1:1** mapping to the reference’s data path and control flow, not new “more clever” patterns the C++ kernel does not use.
2. **QK path in the reference is not `TCvt` on tiles then `TPUSH` from MAT/LEFT.** Cube writes QK to GM with **`TSTORE`** from the **accumulator** (`compute_qk`); vec reads with **`TLOAD`** into vec UB (`compute_p`). There is **no** MAT/LEFT→GM push for QK. Where fp16 appears for **P**, the reference uses the **V2C `TPipe`** / macro path (e.g. `sizeof(half)` slot size in `runTFA`), not a fabricated “fp16 QK wire”.
3. **`TPushOp` accepts ACC (and VEC for the other direction) by design** (`include/PTO/IR/PTOOps.td`). **`PIPE_UNASSIGNED` for MAT/LEFT as `tpush` sources is expected**: there is no direct MAT/LEFT→global path like `TSTORE`. That is **not** a bug to “fix” by widening `tpush` to MAT/LEFT for FA—doing so would **diverge** from the reference.
4. **Allowed toolchain divergence** from hand-written C++: **`ptoas --enable-insert-sync`** (and similar passes) that insert synchronization; everything else should aim at **reference parity**, not new semantics.

Upstream **PTOAS** paths below are **relative to the `PTOAS/` repository root**.

---

## 1. Documentation: reference **QK** path ↔ **`l2g2l_pipe` + ACC `tpush` / `tpop`**

**What the reference does.** Cube **`TSTORE`** of **`float`** tiles shaped **`Cube_S0 × Cube_S1`** into `qk_tile_fifo` with the `base_elems` packing (`fa_kernel.cpp`, `compute_qk`). Vec **`TLOAD`**s **`Vec_S0 × Cube_S1`** slices and assembles **`Vec_S0 × Tile_S1`** in UB (`compute_p`, loop over `sub_col`).

**What the Python builder does today.** Uses **`pto.tpush(qk_acc, qk_pipe)`** with **`slot_size = S0 * S1_TILE * sizeof(fp32)`** and GM backing—**analogous** to getting QK to GM + vec visibility, but **not** identical to the reference’s per-`sub_tile_id` **`Cube_S0×Cube_S1`** store layout when `kTileFactor > 1`.

**Ask (PTOAS / docs only).**

- In `docs/designs/ptoas-tpush-tpop-design.md` (or a small FA note), add a **side-by-side**: reference **`TSTORE`/`TLOAD` + `base_elems`** vs recommended **`slot_size` / `slot_num` / GM pointer math** for `initialize_l2g2l_pipe` so Python authors can **match the reference packing** without guessing.
- Clarify explicitly: **cube `tpush` from ACC** is the supported producer; **do not** document MAT/LEFT `tpush` to GM as a FA workaround—that path **does not exist** in the reference.

---

## 2. **`kTileFactor` / `Cube_S1` K-split and vec `Vec_S0` (reference-only geometry)**

**What the reference does.** `kTileFactor = Tile_S1 / Cube_S1`; two matmul passes per logical tile; vec softmax tile **`Vec_S0 × Tile_S1`** with **`Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor`** (`runTFA`); **`compute_p`** loads **`kTileFactor`** column strips from GM.

**What the Python builder does today.** One **`HEAD × S1_TILE`** matmul and one full **`S0 × S1_TILE`** `tpush`; vec **`tpop`** **`S0_HALF × S1_TILE`** with **`TILE_UP_DOWN`**.

**Ask.**

- **Documented** recipes (and, if useful, **ptodsl** helpers only—no new PTO ops required) for: cube **`AccMode`/`InitPartialSum`/`AccPartialSum`**-style sequences matching `compute_qk` + `compute_pv`, and vec **`pto.load`** / **`slice_view`** patterns matching **`TLOAD`** + **`TASSIGN`** column offsets in `compute_p`.
- If something in **MLIR verification** blocks a **literal** ref-shaped **`pto.store`**/`load` schedule that is otherwise valid, file that as a **narrow bugfix** with a ref citation—not a new feature.

---

## 3. **Software row / subblock indexing (`row_slice`, `get_subblockid`)**

**What the reference does.** `row_offset = subblock_base_rows + row_slice * Vec_S0` and reduce-tile **`TASSIGN`** byte offsets (`compute_p`)—**software** decomposition, not a request for new hardware **`split`** enum values unless the ISA already exposes them for the same pattern.

**Ask.**

- **Examples in docs** showing how to express the same indexing with **existing** Python/PTO constructs (`get_subblock_idx`, scalar offsets, multiple `load`s), so authors do not conflate **`TILE_UP_DOWN`** alone with the reference’s **`row_slice × kTileFactor`** schedule.

---

## 4. **Sync: `TSync_Custom`, CV FIFO depth, `QK_PRELOAD` (exists in reference)**

**What the reference does.** `TSync_Custom<…>` around QK produce / vec consume (`qk2smSync`); `should_wait_consumption` / `should_notify_consumption`; template **`QK_PRELOAD`**, **`qkp_tile_fifo_size`**, **`CV_FIFO_CONS_SYNC_PERIOD`** (`runTFA`, `compute_p`, `compute_qk`).

**What the Python stack does today.** **`--enable-insert-sync`** plus pipe **`tfree`** / implicit ordering—**intentionally** different from hand-placed `TSync_Custom`.

**Ask (reasonable).**

- **Optional** dialect or pass hooks that lower to the **same** sync primitives the reference uses, **or** a documented **equivalence table**: “ref `qk2smSync.wait()` ↔ inserted barrier X after `ptoas` version Y”.
- This is **not** a license to invent new memory paths; it is **parity** for **control** the reference already has.

---

## 5. **Python / `ptodsl` ergonomics (optional; ref-shaped constants)**

**Problem.** Layout math (`GM_*_OFF_F32`, `MAT_*_OFF`, vec FIFO bytes) is hand-rolled and easy to break when `Cube_S0` / `Tile_S1` / `HEAD` change—**the reference avoids some of this** with template parameters and allocator helpers.

**Ask.**

- **Optional** helpers or tables driven only by **`runTFA`-style constants** (`CUBE_S0`, `CUBE_S1`, `TILE_S1`, `QK_PRELOAD`, FIFO sizes) to generate **GM strides matching `base_elems`**, **MAT/VEC base offsets**, and **static overlap checks**—without introducing new runtime algorithms.

---

## 6. **Documentation cross-link (macros)**

**Ask.** In PTOAS or `ptodsl` docs, link **`pto_macro_matmul` / `pto_macro_fa_softmax` / `pto_macro_fa_gu`** sequences to the **minimal** `tile.*` / `pto.*` sequences needed for lowering parity—**macro-internal** ops (including whatever the macro uses for P) are **in scope**; **invented** pre-pipeline **`tile.cvt`** on QK to fake a dtype the reference does not use on that path is **out of scope**.

---

## Concrete examples (reference ↔ Python today)

### A. QK: **`TSTORE`/`TLOAD`** (ref) vs **`tpush`/`tpop`** + ACC (Python)

**Reference — cube writes fp32 QK slices to GM** (`examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp`):

```364:376:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
        using GlobalDataQK =
            GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
        const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
        const size_t base_elems =
            static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) * static_cast<size_t>(Cube_S0) *
                static_cast<size_t>(Cube_S1) +
            static_cast<size_t>(sub_tile_id) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
        GlobalDataQK qkGlobalTile(qk_tile_fifo + base_elems);
        TSTORE(qkGlobalTile, qkAccTile);
```

**Reference — vec reads fp32 from GM into column strips of `qkVecTile`** (same file, `compute_p`):

```505:517:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
        for (int sub_col = 0; sub_col < static_cast<int>(kTileFactor); ++sub_col) {
            __gm__ float *qk_ptr_sub =
                qk_ptr + static_cast<size_t>(sub_col) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
            GlobalDataQK_Sub qkGlobalSub(qk_ptr_sub);

            TileDataF_Sub qkVecSub;
            const uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * Cube_S1 * sizeof(float));
            TASSIGN(qkVecSub, (uint64_t)qkVecTile.data() + col_byte_offset);
            TLOAD(qkVecSub, qkGlobalSub);
        }
```

**Reasonable Python direction (still 1:1 with ref).** Express the same **GM layout + slice loads** using **`pto.store`** / **`pto.load`** + **`slice_view`** on `__gm__` tensors (or keep **`l2g2l_pipe`** but size **`slot_size`** and **`gm_addr`** offsets to match **`base_elems`** / **`kTileFactor`**). **Do not** introduce **`tile.cvt(qk_acc → mat/left fp16)` + `tpush`** for QK: that is **not** in the reference, and MAT/LEFT **`tpush`** is not a supported producer anyway.

**Python builder today** (`experimental/fa_builder.py`): one matmul over full `S1_TILE`, **`pto.tpush(qk_acc[k], qk_pipe)`**, vec **`pto.tpop(qk_vec_ty, …)`** — simpler than ref **`kTileFactor`** loop; **documented** in §1–2 as the gap to close **using ref-shaped stores/loads or matching pipe packing**.

### B. P: **V2C `TPipe` with `sizeof(half)`** (ref) vs vec `tile.cvt` + `tpush_to_aic` (Python)

**Reference — P FIFO slot is fp16-sized, cube pops into MAT** (`fa_kernel.cpp`):

```820:823:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
    using PPipe =
        TPipe<BUF1_SM_READY, Direction::DIR_V2C, Cube_S0 * Cube_S1 * sizeof(half), p_tile_fifo_slots, pMatTNBuffers>;
```

**Python today** uses **`tile.cvt(p_fp32, p_fp16)`** then **`pto.tpush_to_aic(p_fp16, …)`** — that is a **porting of the vec→cube half path**, not the QK cube→vec path. Any **`TCvt`-like behavior for P** should stay aligned with **`pto_macro_fa_softmax`** / reference packing, not used as a precedent to invent QK dtype tricks.

### C. **`PIPE_UNASSIGNED` on non-ACC `tpush`**

If someone tries **`pto.tpush`** from a **MAT** or **LEFT** tile, verification fails (`include/PTO/IR/PTOOps.td`, `TPushOp::getPipe`). That is **consistent** with there being **no** ref-style **`TSTORE`** from MAT/LEFT to GM for QK. **Not a PTOAS feature request for FA**—use **ACC → `tpush`** (pipe) or **`pto.store`** from acc/global views like **`TSTORE`**.

---

## Priority (suggested)

| Priority | Item | Rationale |
|----------|------|-----------|
| P0 | **§1–2** (docs + ref-shaped K/`Vec_S0` lowering recipes) | Largest **semantic** gap vs ref **without** inventing ops. |
| P1 | **§4** (sync parity / equivalence vs `TSync_Custom`) | Exists in ref; **`--enable-insert-sync`** is the only deliberate deviation today. |
| P2 | **§3, §5–6** (indexing examples, ergonomics, macro cross-links) | Reduces author error; no new hardware paths. |

---

## Related artifacts in this repo

- Experimental FA builder: `examples/aot/flash_attention/experimental/fa_builder.py`
- Reference C++ kernel: `examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp`
- Broader gap narrative: `examples/aot/flash_attention/known_gap.md`
