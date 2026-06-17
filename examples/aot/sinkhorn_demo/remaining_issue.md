# Remaining issues: PTODSL Sinkhorn v2 vs hand-written C++ v2

This note captures why **`sinkhorn_v2_builder.py`** (Python / MLIR PTO) does not yet reach the effective bandwidth of **`cpp_ref/kernel_sinkhorn_v2.cpp`**, and what would help close the gap. It is written from the Sinkhorn-K=4 demo (`examples/aot/sinkhorn_demo`).

## Executive summary

The reference C++ v2 kernel wins mainly by (1) **one batched column reduction per Sinkhorn phase** on an **interleaved** UB layout (`TCOLSUM` on a `(K, ROW_BLOCK_COLS)` view), (2) **strided UB copies** for interleave / de-interleave that match that layout exactly, and (3) **large per-chunk work** (double-buffered batch tiles sized from UB budget) with explicit pipeline flags.

The PTODSL v2 large-`N` path today uses a **correct** but more conservative strategy: a **32-matrix stack** with the same row-softmax fusion as `sinkhorn_batch8_builder.py`, plus a **`pto.range` loop over matrices** for column normalize. That avoids a subtle layout bug we hit when combining **manual row permute** with **`tile.reshape`** to an interleaved tile type: numerically the result diverged (NaNs) because the reshape reinterpretation did not match the C++ strided layout byte-for-byte.

Closing the performance gap requires either **proving or encoding** the interleaved memory map in PTODSL, or **new primitives / compiler behavior** below.

---

## Current blockers (Python v2 vs C++ v2)

### 1. No first-class strided UB ↔ UB layout transform in PTODSL

The C++ v2 reference (`cpp_ref/kernel_sinkhorn_v2.cpp`) interleaves / de-interleaves each within-matrix row using **`pto::TCopy`** on **`Tile2D<half, TALL_ROWS, TILE_COLS>`** views (`#include <pto/npu/a2a3/TCopy.hpp>`): `validRow = group_size`, `validCol = K`, and compile-time **half-element row strides** `K * TILE_COLS` (batch) and `TILE_COLS` (WORK). That matches the former `copy_ubuf_to_ubuf` burst pattern without calling **`__builtin_cce_copy_ubuf_to_ubuf`** from the demo kernel.

PTODSL still has no direct equivalent: authors rely on **`tile.mov` on row stripes** and/or **`tile.reshape`**. Reshape is only safe when its linearization is **identical** to the layout produced by the same `TCopy` semantics. A Python binding that lowers to **`TCopy`** (or the same `copy_ubuf_to_ubuf` row loop with explicit strides) would align the interleaved fast path with this reference.

### 2. Batched column normalize vs per-matrix `col_sum` loop

With the safe batch-32 stack, each Sinkhorn phase runs **up to 32 `tile.col_sum` calls** (plus `eps` and `col_expand_div`) instead of **one** `TCOLSUM` over `(K, ROW_BLOCK_COLS)`. On generated code this tends to mean **more vector barriers and less amortized reduction** than C++ v2’s macro (`TCOLSUM` + `TCOLEXPANDDIV` + barriers) on the full interleaved tile.

### 3. Chunk / tile sizing vs UB budget

C++ v2 computes **`MAX_BATCH_ROWS`** from half the UB budget (minus row/col stat regions) and double-buffers, so each **TLOAD** amortizes over many matrices. PTODSL v2 is constrained by **static `TileBufType` shapes**, **`tile.mov` / subview static size rules**, and practical UB limits on 910B. The Python path uses a **fixed 128-row stack (32 matrices)**; the reference can use **substantially larger** batch tiles where memory allows.

### 4. Static subview sizes vs dynamic `valid_row` / chunk length

MLIR `SubViewOp` sizes observed in this demo are **compile-time integers**. Dynamic **`chunk_rows`** / **`chunk_mat`** must be expressed via **`valid_row` on `alloc_tile`** plus **fixed maximum** subview extents (`MAX_BATCH_ROWS`, `K`, etc.). That is workable but limits expressiveness for **generic** “size this tile from SSA budget” patterns the C++ side encodes with template + runtime `min`.

### 5. `pto.tcolsum` / `isBinary` coupling in `ptoas`

Experimentally, lowering **`tile.col_sum(..., is_binary=False)`** failed in `ptoas` with: the temporary operand **requires** the `isBinary` attribute. That blocks exploring alternate reduction semantics where Python might match a different hardware path, and it couples Python surface area to whatever `ptoas` currently accepts for tmp operands.

---

## Feature requests: MLIR PTO Python bindings (`ptodsl`)

These are ordered roughly by impact for the Sinkhorn v2 case; each should come with **documented memory order** (row-major rules, 32-byte alignment, interaction with `valid_row` / `valid_col`).

1. **Strided UB copy (or `tile.copy_ub_strided`)**  
   Expose the same capability as the C++ `stridedUBCopy` / `__builtin_cce_copy_ubuf_to_ubuf` path: `nBurst`, `lenBurst`, `srcGap`, `dstGap`, with tile descriptors for source and destination. Use cases: interleave, de-interleave, and other **non-contiguous** UB reorganizations without guessing reshape linearization.

2. **Layout-tagged reinterpret / “view as” with checked algebra**  
   Either: (a) a **`tile.view_as(other_tile_type, tile, proof_id)`** that is only legal when the compiler proves element count and alignment match, or (b) a **`tile.interleaved_view(tall_tile, K, TILE_COLS, MAX_GROUP_SIZE)`** that returns the `(K, ROW_BLOCK_COLS)` view with the **same** mapping as the reference kernel’s comments (tall row index formula). Goal: eliminate silent mismatch between manual mov loops and reshape.

3. **Fused or batched column reduction API**  
   A single op or macro region that applies **`TCOLSUM`-style reduction + epsilon + `TCOLEXPANDDIV`** over a list of same-shaped subviews, or over a user-described batch of `K×K` tiles in UB, to recover one-barrier amortization similar to C++ v2 without hand-writing MLIR.

4. **Documented interaction: `reshape`, `SubTensorType`, and scratch**  
   Formal rules (and diagnostics) for when **row-reduction scratch** and **column-reduction tmp** may alias the same UB buffer, and how **`tile.reshape`** affects those lifetimes. Today, accidental aliasing showed up as **NaNs** in generated C++; clearer IR invariants or automatic scratch splitting would help.

5. **`pto.range` metadata: unroll / pipeline hints**  
   Optional hints (unroll count, “emit static trip count when bounded by constant `min`”, affinity to reduction ops) so the lowering can match hand-tuned **`#pragma unroll`** and barrier placement in C++ v2’s tail loop.

6. **Dynamic subview sizes where provably bounded**  
   When a SSA value is bounded by a **compile-time constant** (e.g. `chunk_rows <= MAX_BATCH_ROWS` from `min_u(CHUNK, …)`), allow subview sizes to use that **constant upper bound** with **dynamic valid metadata**, reducing the need to always materialize full `MAX_BATCH_ROWS` paths in user code.

---

## Feature requests: `ptoas` (and related lowering)

1. **`pto.tcolsum` tmp operand and `isBinary`**  
   Document when `isBinary` is **required** vs optional for the tmp operand. If non-binary temps are valid on hardware for some reduction modes, allow them and surface **clear errors** when illegal; if they are never valid, document that so Python does not expose a dead `is_binary=False` path.

2. **Scratch allocation / anti-aliasing for reshape + row ops**  
   When analysis detects that **`TRESHAPE`** (or equivalent) overlaps **`TROWSUM` / `TROWMAX`** scratch in UB, **automatically** assign disjoint scratch (or emit a diagnostic with a fix hint), instead of silently aliasing (NaNs at runtime).

3. **Barrier scheduling and reduction fusion**  
   For patterns like “many `TCOLSUM` on disjoint `K×K` subviews of one parent tile in the same phase”, consider **fusing barriers**, **scheduling reductions back-to-back**, or **emitting one wider reduction** when legal—mirroring what a human does with a single interleaved `TCOLSUM` in C++ v2.

4. **Lowering `tile.reshape` to strided copy when contiguous reinterpret is invalid**  
   If the IR cannot prove that reshape is a no-op layout reinterpretation, **refuse** or **lower via explicit copy** with a reported cost estimate, so authors do not rely on accidental correctness.

5. **Optional sync insertion policy per region**  
   Finer control than a global `--enable-insert-sync`: e.g. “minimal barriers for this `vector_section`” vs “safe default”, to experiment with C++-style manual `pipe_barrier` placement without fighting the tool on every edit.

---

## What already works

- The **batch-32 stack** large-`N` path in `sinkhorn_v2_builder.py` matches the reference in **`test_sinkhorn.py`** and improves somewhat over the 8-wide batched PTODSL kernel, but **GB/s remains far below C++ v2** on the same benchmark harness (`bench_sinkhorn_bandwidth.py`).

---

## References in this tree

- PTODSL builder: `sinkhorn_v2_builder.py`  
- Reference kernel and layout comments: `cpp_ref/kernel_sinkhorn_v2.cpp`  
- Generated Ascend C++ (for inspecting barriers and op mix): `outputs/sinkhorn_v2_generated.cpp`
