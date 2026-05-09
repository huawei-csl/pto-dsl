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

**What the Python builder does today (experimental).** **`compute_qk`**: **`kTileFactor`** passes (**`HEAD × Cube_S1`** K slices, **`tile.subview`** on **`qk_acc`** for **`S0 × Cube_S1`** acc columns), then one full **`S0 × S1_TILE`** `tpush`. **`compute_pv`**: still a **single** **`p_left × v_right → pv_acc`** matmul per tile (no K-split / **`AccMode`** striping like the reference yet). Vec **`tpop`** **`S0_HALF × S1_TILE`** with **`TILE_UP_DOWN`**.

**Ask.**

- **Documented** recipes (and, if useful, **ptodsl** helpers only—no new PTO ops required) for: cube **`AccMode`/`InitPartialSum`/`AccPartialSum`**-style sequences matching **`compute_pv`** (and any edge cases in **`compute_qk`**), and vec **`pto.load`** / **`slice_view`** patterns matching **`TLOAD`** + **`TASSIGN`** column offsets in `compute_p`.
- **`tile.subview`** Python API: **`sizes`** must be plain **`int`** (they are forwarded to MLIR `I64ArrayAttr`); **`const()`** wrappers currently fail at build time—document or unwrap in **`tile.subview`**.
- **`compute_pv` K-split in Python:** ref uses **`kTileFactor`** partial matmuls with **P** column strips. **`pto.tmatmul`** requires **`lhs` in LEFT**, but **LEFT** `tile.subview` on default boxed RowMajor **P** rejects column strips (`boxed RowMajor subview must keep full cols`). **MAT** **`p_recv`** allows the same column **`subview`** pattern as **`qk_acc`**, but **MAT cannot be `tmatmul` lhs** today. Reasonable ask: either **document** a supported recipe (e.g. **`TMOV`** staging + layout) or **relax verifier** / **extend `tmatmul`** only where it matches the reference matmul macro contract—not an invented new algorithm.
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

These snippets are for **PTOAS / `ptodsl` authors** mapping hand-written C++ (`fa_kernel.cpp`) to Python builders. Line citations use this repo’s paths.

### A. QK: **`TSTORE`/`TLOAD`** (ref) vs **`tpush`/`tpop`** + ACC (Python)

**Reference — cube writes one `Cube_S0 × Cube_S1` fp32 strip per `sub_tile_id` to GM** (`compute_qk`):

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

**Reference — vec reads `kTileFactor` GM strips and packs them into one wide `qkVecTile`** (`compute_p`):

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

**Python builder today — cube:** `kTileFactor` **inner** matmuls match the ref’s **K** tiling, but the **GM path** is still one **`ACC`** tile **`tpush`** per logical tile (full `S0 × S1_TILE` fp32), not **`kTileFactor`** separate **`TSTORE`** slots.

`accumulate_qk_for_tile` (K strips → **`qk_acc`** column subviews):

```359:376:examples/aot/flash_attention/experimental/fa_builder.py
        def accumulate_qk_for_tile(k_s1_offset, qk_buf, k_mat_buf_idx):
            for sc in range(K_TILE_FACTOR):
                k_col = k_s1_offset + const(sc * CUBE_S1)
                kt_view = pto.slice_view(
                    kt_sub_slice_ty,
                    source=tv_k,
                    offsets=[c0, k_col],
                    sizes=[cHEAD, cCUBE_S1],
                )
                pto.load(kt_view, k_mat[k_mat_buf_idx])
                tile.mov(k_mat[k_mat_buf_idx], k_right[k_mat_buf_idx])
                qk_part = tile.subview(
                    qk_acc[qk_buf],
                    [c0, const(sc * CUBE_S1)],
                    [S0, CUBE_S1],
                )
                tile.matmul(q_left, k_right[k_mat_buf_idx], qk_part)
```

Prologue: one **`tpush`** per logical tile after **`accumulate_qk_for_tile`**:

```387:391:examples/aot/flash_attention/experimental/fa_builder.py
            for k in range(QK_PRELOAD):
                k_s1_off = const(k * S1_TILE)
                accumulate_qk_for_tile(k_s1_off, k, k % 2)
                pto.tpush(qk_acc[k], qk_pipe, SPLIT_UP_DOWN)
```

**Python builder today — vec:** one **`tpop`** of a **half-tile** `qk_vec_ty` (`S0_HALF × S1_TILE` fp32) per softmax step, not **`Vec_S0 × Cube_S1`** repeated **`TLOAD`**s:

```571:605:examples/aot/flash_attention/experimental/fa_builder.py
        def emit_softmax_step(exp_max_slot, is_init):
            qk_recv = pto.tpop(
                qk_vec_ty,
                qk_pipe,
                SPLIT_UP_DOWN,
                addr=const(VEC_RECV_OFF, s.int64),
            )
            tile.muls(qk_recv, scale, qk_recv)
            tile.row_max(qk_recv, p_fp32, local_max)
            # ... softmax body ...
            tile.cvt(p_fp32, p_fp16)
            pto.tpush_to_aic(p_fp16, SPLIT_UP_DOWN, id=ID_P)
            pto.tfree(qk_pipe, SPLIT_UP_DOWN)
```

**Still missing vs ref (PTOAS / docs ask).** Either **document** how **`slot_size` / GM offsets** should reproduce **`base_elems` + `sub_tile_id`** packing, or show **`pto.store`/`pto.load`** that mirror **`TSTORE`/`TLOAD`** exactly. **Do not** document **`tile.cvt` on QK + `tpush` from MAT/LEFT`**; that is not the reference QK path.

### A2. Schedule: **`runTFA`’s `sub_tile` loop** (C++) vs **fused Python steps** (same semantics, different shape)

The reference **interleaves** cube **`compute_qk`** and vec **`compute_p`** at **`kTileFactor`** granularity inside the steady loop:

```848:920:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
    for (int preload_tile = 0; preload_tile < static_cast<int>(qkPreloadNum) && preload_tile < num_tiles_s1;
         ++preload_tile) {
        if constexpr (DAV_CUBE) {
            for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor); ++sub_tile) {
                compute_qk<...>(preload_tile, sub_tile, ...);
            }
        }
        if constexpr (DAV_VEC) {
            for (int row_slice = 0; row_slice < static_cast<int>(kTileFactor); ++row_slice) {
                compute_p<...>(preload_tile, row_slice, ...);
            }
        }
    }

    for (int tile_id = 0; tile_id < num_tiles_s1; ++tile_id) {
        // ...
        for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor); ++sub_tile) {
            if constexpr (DAV_CUBE) {
                if (next_qk_tile != -1) {
                    compute_qk<...>(next_qk_tile, sub_tile, ...);
                }
            }
            if constexpr (DAV_VEC) {
                if (next_qk_tile != -1) {
                    compute_p<...>(next_qk_tile, sub_tile, ...);
                }
            }
            if constexpr (DAV_CUBE) {
                compute_pv<...>(tile_id, sub_tile, ...);
            }
        }
        if constexpr (DAV_VEC) {
            compute_gu<...>(tile_id, ...);
        }
    }
```

**Python** (`experimental/fa_builder.py`) **fuses** all **`kTileFactor`** QK matmuls **inside** `accumulate_qk_for_tile` before **`tpush`**, and vec **`emit_softmax_step`** does **not** take explicit **`row_slice`** or **`sub_col`** arguments—**`TILE_UP_DOWN`** replaces part of the ref’s **`row_slice × Vec_S0`** story. A **PTOAS-facing doc** should spell out: “**`row_slice` loop** ↔ **`get_subblock_idx()` + fixed `S0_HALF`**” and “**`sub_col` GM loads** ↔ **either** replicated **`slice_view`** **or** one wide **`tpop`**,” so reviewers do not assume bit-identical control flow.

### B. P: **V2C `TPipe` with `sizeof(half)`** (ref) vs vec **`tile.cvt`** + **`tpush_to_aic`** (Python)

**Reference — P FIFO slot is fp16-sized, cube pops into MAT** (`fa_kernel.cpp`):

```820:823:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
    using PPipe =
        TPipe<BUF1_SM_READY, Direction::DIR_V2C, Cube_S0 * Cube_S1 * sizeof(half), p_tile_fifo_slots, pMatTNBuffers>;
```

**Python — same logical edge (vec → cube), different primitive names:**

```603:604:examples/aot/flash_attention/experimental/fa_builder.py
            tile.cvt(p_fp32, p_fp16)
            pto.tpush_to_aic(p_fp16, SPLIT_UP_DOWN, id=ID_P)
```

**Cube consumer** still **`tpop`**s into **`p_recv_ty`** (MAT), then **`mov`** to **`p_left`** for **`matmul`**—analogous to **`TPOP`** into **`pMatTile`**. The **ask for PTOAS** is macro-order parity (**`pto_macro_fa_softmax`**) and **slot byte size** documentation, not new QK **`TCvt`** ideas.

### C. **`PIPE_UNASSIGNED` on non-ACC `tpush`**

**What fails today (do not suggest as FA “fix”):**

```python
# Hypothetical / INVALID for QK in current PTO lowering:
pto.tpush(some_left_tile, qk_pipe, split=1)   # LEFT → pipe: not a supported QK producer
pto.tpush(some_mat_tile, qk_pipe, split=1)    # MAT → pipe: verifier / PIPE_UNASSIGNED
```

**What the Python FA builder uses instead (supported):**

```python
pto.tpush(qk_acc_tile, qk_pipe, SPLIT_UP_DOWN)  # ACC → l2g2l pipe (supported producer)
```

That matches the **intent** of ref **`TSTORE`** from **accumulator** data to a **visibility** buffer, even though the **packing** still differs from per-**`sub_tile_id`** **`TSTORE`** (§A).

### D. PV: **`AccMode` + `kTileFactor`** in C++ vs **single `matmul`** in Python (verifier gap)

**Reference — outer caller invokes `compute_pv` once per `sub_tile`** (`runTFA`):

```912:918:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
            if constexpr (DAV_CUBE) {
                compute_pv<HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, pv_tile_fifo_size,
                           CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK>(
                    tile_id, sub_tile, v, pv_tile_fifo_block, pMatTile[pv_src_pingpong_id % pMatTNBuffers],
                    vMatTile[pv_src_pingpong_id % vMatTNBuffers], pvAccTile,
                    pv_src_pingpong_id % vMatTNBuffers + PV_EVENT_ID0, pvAccTileEvtID, pPipe, pv2guSync);
                pv_src_pingpong_id++;
            }
```

**Reference — inner `AccMode` chooses init vs accumulate across K strips** (`compute_pv`):

```400:433:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
        const int s1_index = tile_id * static_cast<int>(Tile_S1) + sub_tile_id * static_cast<int>(Cube_S1);
        // ...
        GlobalVT vLoad((__gm__ half *)(v + s1_index * HEAD_SIZE));
        TLOAD(vMatTile, vLoad);

        TPOP<PPipe, TileMatPData, TileSplitAxis::TILE_UP_DOWN>(pPipe, pMatTile);

        const AccMode accMode = (sub_tile_id == 0) ?
                                    (is_last_subtile ? AccMode::InitFinalSum : AccMode::InitPartialSum) :
                                    (is_last_subtile ? AccMode::AccFinalSum : AccMode::AccPartialSum);
        pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD>(pMatTile, vMatTile, pvAccTile, accMode);
```

**What a literal Python port would look like** (conceptual; **not** all verifiable today):

```python
# Desired shape: P is MAT, V is MAT, pv is ACC — ref uses pMatTile strips × v strips.
for sc in range(K_TILE_FACTOR):
    p_sub = tile.subview(p_mat, [0, sc * CUBE_S1], [S0, CUBE_S1])       # column strip of P on MAT
    v_sub = tile.subview(v_right, [sc * CUBE_S1, 0], [CUBE_S1, HEAD])  # row strip of V on RIGHT
    if sc == 0:
        tile.matmul(p_sub, v_sub, pv_acc)
    else:
        tile.matmul_acc(pv_acc, p_sub, v_sub, pv_acc)
```

**What actually blocks this in `ptodsl` today:**

- **`tile.subview(p_left, …, [S0, CUBE_S1])`** (column strip on default boxed **LEFT** RowMajor **P**) → MLIR: **`boxed RowMajor subview must keep full cols`**.
- **`tile.matmul(p_sub_mat, v_sub, …)`** with **`p_sub_mat`** on **MAT** → MLIR: **`tmatmul` expects lhs in LEFT`**.

So the **missing PTOAS / dialect / doc** piece is a **supported** way to express **ref `pto_macro_matmul` + `AccMode`** for **PV**—either **documented staging** (**`TMOV`** MAT→LEFT strips with a legal layout, or **`tmatmul`** generalization **only** where it matches the macro contract), not a new FA algorithm.

**Python steady-state PV today** (one full matmul, ref-equivalent **math**, different **micro-schedule**):

```408:423:examples/aot/flash_attention/experimental/fa_builder.py
                p_raw = pto.tpop_from_aiv(p_recv_ty, SPLIT_UP_DOWN, id=ID_P)
                tile.mov(p_raw, p_left[b])
                pto.tfree_from_aiv(SPLIT_UP_DOWN, id=ID_P)
                tile.mov(v_mat[b], v_right[b])
                # ... prefetch next V into v_mat[1 - b] ...
                tile.matmul(p_left[b], v_right[b], pv_acc[b])
                pto.tpush(pv_acc[b], pv_pipe, SPLIT_UP_DOWN)
```

### E. **`tile.subview` sizes: Python `int` vs `s.const` (MLIR `I64ArrayAttr`)**

**Fails at Python build time** (`TypeError` from `IntegerAttr.get`):

```python
cS0 = const(S0)
cCUBE_S1 = const(CUBE_S1)
qk_part = tile.subview(qk_acc, [c0, const(sc * CUBE_S1)], [cS0, cCUBE_S1])  # BAD: sizes are Value wrappers
```

**Works** (sizes are plain integers; offsets can stay dynamic via **`const`** / scalars as today):

```python
qk_part = tile.subview(qk_acc, [c0, const(sc * CUBE_S1)], [S0, CUBE_S1])
```

**Ask:** either **unwrap** **`const`** in **`ptodsl.api.tile.subview`** for **`sizes`**, or **document** this in PTOAS / `ptodsl` API reference so FA-style builders do not rediscover it via traceback.

### F. Sync: **`TSync_Custom` + `wait`/`allocate`/`record`/`free`** (C++) vs **pipes + `tfree`** (Python)

**Reference — explicit producer/consumer pairing on the QK path** (`compute_qk` / `compute_p`):

```361:381:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
        if (sub_tile_id == 0 && should_wait_consume)
            qk2smSync.allocate(); // wait for SM consume data
        // ...
        TSTORE(qkGlobalTile, qkAccTile);
        // ...
        if (sub_tile_id == static_cast<int>(kTileFactor) - 1)
            qk2smSync.record(); // notify for QK produce data
```

```496:520:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
        wait_flag(PIPE_V, PIPE_MTE2, pTileEventId);
        if (row_slice == 0)
            qk2smSync.wait(); // wait for QK produce data
        // ... TLOAD kTileFactor strips ...
        if (row_slice == static_cast<int>(kTileFactor) - 1 && should_notify_consume)
            qk2smSync.free(); // notify for SM consume data
```

**Sync object type** (template parameter baked into `runTFA`):

```817:817:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> qk2smSync = {BUF0_QK_READY};
```

**Python — no named `TSync_Custom`; ordering relies on pipe ops + toolchain-inserted sync** (`--enable-insert-sync` in `experimental/compile.sh`):

```279:281:examples/aot/flash_attention/experimental/fa_builder.py
        qk_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_QK,
```

**Ask for PTOAS / docs:** a small **equivalence table** row per ref hook, e.g. **`qk2smSync.record()` after last `TSTORE` of a tile** ↔ **which `tpush` / GM completion barrier** in generated C++ after `ptoas` version *X*, so performance regressions can be bisected without reading all inserted barriers.

### G. Row / column indexing: **`row_slice` + `Vec_S0`** (C++) vs **`get_subblock_idx` + `S0_HALF`** (Python)

**Reference — software row origin per vec core and `row_slice`** (`compute_p`):

```488:492:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
        const size_t subblock_base_rows =
            static_cast<size_t>(Cube_S0 / VEC_CORES) * static_cast<size_t>(get_subblockid());
        const size_t row_offset = subblock_base_rows + static_cast<size_t>(row_slice * Vec_S0);
        const int s0_index = blk_idx * Cube_S0 + row_offset;
```

**Python — subblock row origin** (half of **`Cube_S0`** per AIV sub-block for **`TILE_UP_DOWN`**):

```540:541:examples/aot/flash_attention/experimental/fa_builder.py
        sb_idx = s.index_cast(pto.get_subblock_idx())
        row_off_sb = sb_idx * cS0_HALF
```

There is **no** Python **`for row_slice in range(kTileFactor):`** around softmax; **`row_slice × Vec_S0`** from the ref is only partially reflected via **`S0_HALF`** recv tiles + pipe **`split`**. The **missing documentation** is a **direct mapping table** (`row_slice`, `Vec_S0`, `kTileFactor`) ↔ (`TILE_UP_DOWN`, `S0_HALF`, `pto.tpop` tile types), so authors do not treat **`split=1`** as a complete substitute for **ref softmax tile decomposition** without proof.

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
