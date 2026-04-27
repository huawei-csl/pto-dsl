# PTOAS feature requests (PTO MLIR dialect + Python bindings)

This document collects **actionable requests** for the PTOAS / PTO dialect stack so that **flash-attention–style kernels** written in Python (e.g. `examples/aot/flash_attention/experimental/fa_builder.py` via `ptodsl`) can **closely match** the hand-tuned reference in `examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp` (`runTFA`, `compute_qk`, `compute_p`, `compute_pv`, `compute_gu`).

Upstream **PTOAS** sources below use paths **relative to the `PTOAS/` repository root** (same layout as a normal PTOAS checkout), e.g. `include/PTO/IR/PTOOps.td`, `docs/designs/ptoas-tpush-tpop-design.md`.

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

## Concrete examples (reference C++ ↔ desired Python ↔ today)

Each subsection ties **one reference pattern** to **what Python would ideally emit**, what **PTOAS / MLIR rejects or cannot express**, and what **`experimental/fa_builder.py` does instead**.

### A. fp16 payload on the QK cube→vec path (requests **1** and **2**)

**Reference (GM is fp32; vec uses narrower working tiles and fp16 for P).** Cube stores each **`Cube_S0 × Cube_S1`** QK slice as **float** in `qk_tile_fifo` (not a hardware `TPUSH` from vec’s perspective—MTE `TSTORE` to GM):

```364:381:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
        using GlobalDataQK =
            GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
        const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
        const size_t base_elems =
            static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) * static_cast<size_t>(Cube_S0) *
                static_cast<size_t>(Cube_S1) +
            static_cast<size_t>(sub_tile_id) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
        GlobalDataQK qkGlobalTile(qk_tile_fifo + base_elems);

#if UF_ENABLE
        TSTORE<STPhase::Final>(qkGlobalTile, qkAccTile);
#else
        TSTORE(qkGlobalTile, qkAccTile);
        set_flag(PIPE_FIX, PIPE_M, accTileEvtID);
#endif

        if (sub_tile_id == static_cast<int>(kTileFactor) - 1)
            qk2smSync.record(); // notify for QK produce data
```

The **P** pipe uses **`Cube_S0 * Cube_S1 * sizeof(half)`** slots (fp16 on the vec→cube wire):

```820:823:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
    constexpr uint32_t p_tile_fifo_slots = qkp_tile_fifo_size * kTileFactor;
    using PPipe =
        TPipe<BUF1_SM_READY, Direction::DIR_V2C, Cube_S0 * Cube_S1 * sizeof(half), p_tile_fifo_slots, pMatTNBuffers>;
    PPipe pPipe((__gm__ void *)p_tile_fifo_block, 0u, (uint32_t)(uint64_t)pMatTile[0].data());
```

**Desired Python pattern (sketch).** After `tile.matmul(..., qk_acc)`, narrow the C2V **`slot_size`** while keeping matmul in fp32:

```python
# Ideal: half wire, same logical tile id
tile.cvt(qk_acc, qk_half_tile, rmode="round")  # TileBufType(..., dtype=f16, memory_space="MAT"|"LEFT")
pto.tpush(qk_half_tile, qk_pipe, SPLIT_UP_DOWN)
```

Vec would `tpop` into **`!pto.tile_buf<vec, … x f16>`** and `tile.cvt` to fp32 before `row_max`.

**Current failing behavior.** `TPushOp::getPipe()` in upstream PTO only treats **`ACC`** (and **`VEC`**) as having a real pipe id; **`MAT` / `LEFT` / …** fall through to **`PIPE_UNASSIGNED`**, so MLIR verification fails:

```1767:1793:include/PTO/IR/PTOOps.td
    ::mlir::pto::PIPE getPipe() {
      auto getAddressSpace = [](Type ty) -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        // ...
      };

      auto as = getAddressSpace(getTile().getType());
      if (!as)
        return ::mlir::pto::PIPE::PIPE_UNASSIGNED;
      if (*as == ::mlir::pto::AddressSpace::ACC)
        return ::mlir::pto::PIPE::PIPE_FIX;
      if (*as == ::mlir::pto::AddressSpace::VEC)
        return ::mlir::pto::PIPE::PIPE_MTE3;
      return ::mlir::pto::PIPE::PIPE_UNASSIGNED;
    }
```

Typical diagnostic: **`'pto.tpush' op tile type must map to a supported producer pipe`**.

**Un-optimal workaround in `fa_builder.py`.** Keep **`SLOT_SIZE_QK = S0 * S1_TILE * 4`** and push **only** the fp32 accumulator (legal **`ACC`** producer):

```74:77:examples/aot/flash_attention/experimental/fa_builder.py
# Per-pipe slot sizes (bytes).
# QK: fp32 on the wire (cube `tpush` only accepts ACC tiles today — see known_gap).
SLOT_SIZE_QK = S0 * S1_TILE * 4
SLOT_SIZE_PV = S0 * HEAD * 4  # fp32 PV accumulator
```

```361:362:examples/aot/flash_attention/experimental/fa_builder.py
                tile.matmul(q_left, k_right[k], qk_acc[k])
                pto.tpush(qk_acc[k], qk_pipe, SPLIT_UP_DOWN)
```

That **doubles** ring bytes versus a half-precision wire format with the same logical geometry.

---

### B. `kTileFactor` / `Vec_S0` vs one big matmul + one `tpop` (requests **3** and **4**)

**Reference geometry.** `runTFA` fixes **`kTileFactor = Tile_S1 / Cube_S1`** and **`Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor`** (e.g. **32** row softmax tile height when **`Cube_S0 = 128`**, **`kTileFactor = 2`**):

```680:691:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
    constexpr uint32_t Cube_S0 = CUBE_S0;
    uint32_t block_rows = s0 / Cube_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1; // per-tile S1 chunk
    constexpr uint32_t Tile_S1 = TILE_S1; // logical tile along S1
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by Cube_S1");
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1; // sub-tiles per TILE_S1
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
    constexpr uint32_t VecGuRows = Cube_S0 / VEC_CORES;
    static_assert(Cube_S0 % (VEC_CORES * kTileFactor) == 0, "Vec rows must divide evenly across tile slices");
```

Vec **softmax** tile type is **`Vec_S0 × Tile_S1`**, not **`Cube_S0/2 × Tile_S1`**:

```747:751:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
    // Define tile types for FA softmax P computation
    // UB offsets for softmax tiles
    // Define per-tile vector tiles sized to Cube_S1
    using TileDataF_T = Tile<TileType::Vec, float, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Tile_S1>;
    using TileDataH_T = Tile<TileType::Vec, half, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Tile_S1>;
```

**Reference assembly of the wide QK tile from K-slices in GM** (`compute_p`): two **`TLOAD`**s of **`Vec_S0 × Cube_S1`** into column halves of **`qkVecTile`**:

```500:517:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
        const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
        const size_t base_elems = static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) *
                                  static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
        __gm__ float *qk_ptr = qk_tile_fifo + base_elems + row_offset * static_cast<size_t>(Cube_S1);

        using GlobalDataQK_Sub =
            GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
        using TileDataF_Sub = Tile<TileType::Vec, float, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Cube_S1>;
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

**Desired Python pattern (sketch).** Mirror **`compute_qk`**’s **`sub_tile_id`** loop with **`AccMode`/`InitPartialSum`** semantics, **`slot_size`/`slot_num`** matching **`base_elems`**, and vec **`tpop`** / **`load`** into **`Vec_S0 × S1_TILE`** (or explicit subview column packing) instead of one **`S0_HALF × S1_TILE`** receive tile per hardware half.

**Current behavior.** The Python builder performs **one** `matmul` over **`HEAD × S1_TILE`** per logical tile and **one** `tpush` of the full **`S0 × S1_TILE`** accumulator; vec uses **`qk_vec_ty` shape `[S0_HALF, S1_TILE]`** with **`TILE_UP_DOWN`**:

```350:362:examples/aot/flash_attention/experimental/fa_builder.py
            for k in range(QK_PRELOAD):
                k_off = const(k * S1_TILE)
                kt_view_k = pto.slice_view(
                    kt_sub_ty,
                    source=tv_k,
                    offsets=[c0, k_off],
                    sizes=[cHEAD, cS1_TILE],
                )
                pto.load(kt_view_k, k_mat[k])
                tile.mov(k_mat[k], k_right[k])
                tile.matmul(q_left, k_right[k], qk_acc[k])
                pto.tpush(qk_acc[k], qk_pipe, SPLIT_UP_DOWN)
```

```550:556:examples/aot/flash_attention/experimental/fa_builder.py
        def emit_softmax_step(exp_max_slot, is_init):
            qk_recv = pto.tpop(
                qk_vec_ty,
                qk_pipe,
                SPLIT_UP_DOWN,
                addr=const(VEC_RECV_OFF, s.int64),
            )
```

There is **no** first-class equivalent to **`row_slice` × `sub_col` × `TASSIGN` column packing** in this path.

**Un-optimal workarounds.**

- **Omit `kTileFactor` on cube** (single large K tile): simpler schedule but **not** the reference’s **`CUBE_S1 = 128`** matmul shape / partial-sum story.
- **Accept `S0_HALF = S0 // 2` vec rows per `tpop`**: matches **`TILE_UP_DOWN`** hardware split, but **not** the reference’s **`Vec_S0 = Cube_S0 / (2 * kTileFactor)`** (e.g. **64×256** received per subblock vs ref **32×256** working tile).

---

### C. `QK_PRELOAD == 4` and explicit producer/consumer sync (requests **5** and **6**)

**Reference preload and sync objects.** The launch uses **`qkPreloadNum = QK_PRELOAD`** (template parameter), **`TSync_Custom`** between cube **`TSTORE`** and vec **`TLOAD`**, and nested **`kTileFactor`** loops so cube and vec each run **`kTileFactor`** steps per logical preload tile:

```817:874:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> qk2smSync = {BUF0_QK_READY};
    constexpr TSync_Custom<SyncOpType::TSTORE_C2GM, SyncOpType::TLOAD> pv2guSync = {UPDATE_READY};
    // ...
    for (int preload_tile = 0; preload_tile < static_cast<int>(qkPreloadNum) && preload_tile < num_tiles_s1;
         ++preload_tile) {
        if constexpr (DAV_CUBE) {
            for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor); ++sub_tile) {
                qkAccTileEvtID = assign_running_acc_tile(qkAccTile);
                compute_qk<HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, CV_FIFO_CONS_SYNC_PERIOD,
                           INTERMEDIATE_CHECK, CAUSAL_MASK>(preload_tile, sub_tile, q_block, k, qk_tile_fifo_block,
                                                            qMatTile[0], kMatTile[k_src_pingpong_id % kMatTNBuffers],
                                                            qkAccTile, k_src_pingpong_id % kMatTNBuffers,
                                                            qkAccTileEvtID, qk2smSync, block_idx);
                k_src_pingpong_id++;
            }
        }
        if constexpr (DAV_VEC) {
            for (int row_slice = 0; row_slice < static_cast<int>(kTileFactor); ++row_slice) {
                compute_p<HEAD_SIZE, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, CV_FIFO_CONS_SYNC_PERIOD,
                          INTERMEDIATE_CHECK, CAUSAL_MASK>(
                    preload_tile, row_slice, qk_tile_fifo_block, exp_max_ififo_block, global_sum_block, exp_max_block,
                    qkVecTile[p_gu_src_pingpong_id % srcVecTNBuffers], x_expT[p_gu_src_pingpong_id % xexpVecTNBuffers],
                    x_expPushT,
                    input_reduce_tmp, m1_local_max, l1_local_sum, m2_global_max, l2_global_sum,
                    l1_exp_max_ififo[preload_tile % qkp_tile_fifo_size], triu, p_gu_src_pingpong_id % xexpVecTNBuffers,
                    qk2smSync, pPipe, block_idx);
                p_gu_src_pingpong_id++;
            }
        }
    }
```

Inside **`compute_p`**, vec **`qk2smSync.wait()`** / **`free()`** bracket GM visibility:

```496:520:examples/aot/flash_attention/cpp_ref/naive_tpush/fa_kernel.cpp
        wait_flag(PIPE_V, PIPE_MTE2, pTileEventId);
        if (row_slice == 0)
            qk2smSync.wait(); // wait for QK produce data
        // ... TLOAD assembly ...
        if (row_slice == static_cast<int>(kTileFactor) - 1 && should_notify_consume)
            qk2smSync.free(); // notify for SM consume data
```

**Desired Python pattern.** Set **`QK_PRELOAD = 4`**, model **`l1_exp_max_ififo[qkp_tile_fifo_size]`**, and emit **named sync** (or dialect ops that lower like **`TSync_Custom`**) aligned with **`should_wait_consumption` / `should_notify_consumption`** from the reference.

**Current behavior / limits.**

- **`initialize_l2g2l_pipe`** requires **`local_slot_num > 0`** and a **peer `reserve_buffer`**; vec UB scales with **`FIFO_BYTES_QK ≈ slot_size × local_slot_num`**. There is **no** built-in “vec only **`TLOAD`** from `gm_addr` after cube slot closes” mode with **zero** local ring bytes.
- Python FA relies on **`ptoas --enable-insert-sync`** for cross-kernel ordering instead of **`TSync_Custom`**-style explicit tokens.

**Un-optimal workaround in `fa_builder.py`.** Hard-code **`QK_PRELOAD = 2`** and a **two-tile `exp_max` ring**; document that raising preload needs more UB and/or hazard rework:

```65:68:examples/aot/flash_attention/experimental/fa_builder.py
# QK preload depth — must be >= 1; reference launch uses 4, this builder
# keeps 2 for a smaller VEC exp_max ring (see header comment).
# (NUM_TILES - QK_PRELOAD) must be even — steady state is pair-unrolled.
QK_PRELOAD = 2
```

---

### D. Manual UB / MAT layout bookkeeping (request **7**)

**Reference** hides much of this behind **`allocate_cube_tile_buffers` / `allocate_vec_tile_buffers`** templates (`runTFA`).

**Desired Python / bindings.** Declarative **“tile name → (space, dtype, shape)”** map that **checks overlaps** and derives **`MAT_P_FIFO_OFF`**, vec FIFO bases, and recv scratch **automatically** when `Cube_S0`, `Tile_S1`, or `HEAD` change.

**Current behavior.** Authors must align **`reserve_buffer`**, **`import_reserved_buffer`**, **`alloc_tile`**, and **`initialize_l2g2l_pipe.slot_size`** by hand.

**Concrete workarounds in `fa_builder.py` today.**

- **Pad `MAT_P_FIFO_OFF`** so the P V2C FIFO cannot overlap growing MAT tiles when **`S0`** increases:

```100:109:examples/aot/flash_attention/experimental/fa_builder.py
# Explicit local-memory layout used when compiling with --pto-level=level3.
# Offsets are byte offsets within each independent local address space.
MAT_Q_OFF = 0
MAT_K_OFF = MAT_Q_OFF + S0 * HEAD * 2
MAT_P_RECV_OFF = MAT_K_OFF + HEAD * S1_TILE * 2
MAT_V_OFF = MAT_P_RECV_OFF + S0 * S1_TILE * 2
MAT_P_FIFO_OFF = MAT_V_OFF + S1_TILE * HEAD * 2
# Pad past the last MAT-resident tile; bisheng is sensitive to overlap here.
if MAT_P_FIFO_OFF < 393216:
    MAT_P_FIFO_OFF = 393216
```

- **Shrink reduce-tile ring footprint** with a computed stride instead of a fixed **512** bytes per slot:

```126:127:examples/aot/flash_attention/experimental/fa_builder.py
# Tight packing for reduce / exp_max ring scalars (one column per logical row).
VEC_RED_STRIDE = ((S0_HALF * 4 + 127) // 128) * 128
```

- **Reuse `p_fp32` as `row_max` scratch** so `tmp_tile` remains free for **`row_sum`**, saving one large vec buffer’s worth of peak live data:

```557:558:examples/aot/flash_attention/experimental/fa_builder.py
            tile.muls(qk_recv, scale, qk_recv)
            tile.row_max(qk_recv, p_fp32, local_max)
```

- **Single `VEC_RECV_OFF` scratch** sized for the max of half-tile QK (if ever narrowed) and half-tile PV:

```134:136:examples/aot/flash_attention/experimental/fa_builder.py
# Shared recv scratch: max(fp16 QK half-tile, fp32 PV half-tile) for tpop addr=.
_VEC_RECV_BYTES = max(S0_HALF * S1_TILE * 2, S0_HALF * HEAD * 4)
VEC_RECV_OFF = VEC_RED_BASE_OFF + 6 * VEC_RED_STRIDE
```

These are **correctness-preserving micro-optimizations**; they do **not** replace dialect support for **§A–§C**.

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
