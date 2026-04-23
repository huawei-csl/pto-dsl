# Flash-Attention DSL kernel — feature gaps vs. `fa_performance_kernel.cpp`

This folder contains a multi-pipe, software-pipelined Flash-Attention kernel
written in pto-dsl. It currently reaches **0.94× of `npu_fused_infer_attention_score`**
at `(Q_ROWS=2048, S1=8192, HEAD=32)` (185 µs vs 175 µs).

## Usage

```bash
bash ./compile.sh
python ./run.py
```


A direct port of the reference `fa_performance_kernel.cpp` (a2a3) at its native
shape `(HEAD=128, TILE_S1=256, CUBE_S1=128, QK_PRELOAD=4)` is currently
**impossible to express in the DSL**. This document lists the concrete
ptoas / pto-dsl features needed to close the gap, ordered by impact.

---

## 1. Sub-tile views on L1/L0/UB `TileBuf` — **biggest blocker**

**Need:** a `pto.tile_subview(tile, offsets, sizes)` op returning a strided view
into a sub-region of an existing `TileBuf` (analogous to `pto.slice_view` for GM
tensors), plus a `tile.matmul` overload accepting a subview as its `out` operand.

**What it unlocks:** the reference's `kTileFactor = TILE_S1 / CUBE_S1 = 2`
decomposition — one wide ACC tile filled by N narrower matmuls writing into
column slices, then one push to vec:
```cpp
TileBuf<S0, 256, fp32, ACC> qk_acc;
matmul(q_left, k_right_lo, qk_acc[:, 0:128]);     // needs subview
matmul(q_left, k_right_hi, qk_acc[:, 128:256]);   // needs subview
tpush(qk_acc, qk_pipe);                           // single 256-wide push
```
Without it our cube-side `S1_TILE` is forced to equal `CUBE_S1`, so vec pays
its per-tile overhead (softmax setup, FFTS acks, pipe push/pop sync) twice as
often as the reference.

**Underlying machinery:** `memref.subview` already works on
`#pto.address_space<acc>` / `<vec>` / `<mat>` in MLIR — the codegen path exists.
Only the Python wrapper + matmul-into-subview support are missing.

**Expected impact:** ~2× vec amortization. Closes the largest single piece of
the perf gap.

---

## 2. Configurable consumer-ack period (`kFaCvFifoConsSyncPeriod`)

**Need:** add a `cons_sync_period: int = 1` kwarg to
`pto.aic_initialize_pipe`, `pto.aiv_initialize_pipe`, and
`pto.initialize_l2g2l_pipe`. Forward it to the corresponding ptoas init op
attribute.

**What it unlocks:** batched FFTS consumer acks (reference uses period=4 →
one ack per 4 pops instead of one per pop). Hardware FFTS already supports it;
only the API surface is missing.

**Effort:** ~10 lines of plumbing in `ptodsl/api/pto.py` + matching MLIR op
attribute. **No compiler-pass changes.**

**Expected impact:** ~15–25% sync-overhead reduction.

---

## 3. Decouple `local_slot_num` from `slot_num` on legacy pipe init

**Need:** the V→C-direction pipes (`aic_initialize_pipe` / `aiv_initialize_pipe`)
hard-code `local_slot_num = slot_num`, forcing a `SLOT_NUM = 8` local FIFO
(e.g. 32 KB of MAT for the P pipe, even when 1 local slot would suffice).
`initialize_l2g2l_pipe` already accepts `local_slot_num` — that pattern needs
to be available in the V→C direction too.

**Two paths:**
- **(a) easy:** add `local_slot_num` kwarg to the legacy ops and stop hard-coding.
- **(b) cleaner:** make `initialize_l2g2l_pipe` work in **both** directions
  (V→C as well as C→V) and deprecate the legacy ops. Requires ptoas C++ work
  to unify the V→C sync model.

**Expected impact:** frees ~28 KB of MAT, enables larger per-tile budgets.

---

## 4. Deeper exp_max ring / `QK_PRELOAD ≥ 4` (auto-sync event-pool exhaustion)

**Symptom:** with `QK_PRELOAD=4` + a 4-deep exp_max ring + ping-pong, the kernel
**deadlocks at runtime**. The `--enable-insert-sync` pass exhausts its
8-event pool on `PIPE_V ↔ PIPE_MTE2` and emits a `wait_flag` whose matching
`set_flag` was reused by a later op.

**Root cause:** the auto-sync pass allocates event IDs round-robin per
`tpush`/`tpop` site without doing liveness analysis on tile-buffer SSA values.

**Three possible fixes (in order of preference):**
- **(c) bump pool size:** the a3 hardware exposes 16 event IDs per direction;
  the pass artificially uses 8. **Possibly a one-line constant change in ptoas**
  — would unlock `QK_PRELOAD=4` immediately.
- **(a) liveness-aware allocator:** rewrite the event-ID allocator in
  `--enable-insert-sync` to be aware of tile-buffer reuse. Significant work.
- **(b) explicit escape hatch:** add `event_id: Optional[int]` kwarg on
  `tpush`/`tpop` for users who want to manage the pool manually.

**Expected impact:** ~10–15% from increased cube/vec drift window.
**Please investigate (c) first — likely a quick win.**

---

## 5. Multi-K fused matmul

**Need:** new op `pto.tmatmul_chain([lhs0, rhs0, lhs1, rhs1, ...], out)`
or compiler fusion of adjacent `pto.tmatmul` ops with same `dst` inside an
`scf.for`. Hardware MMA chains K-subtiles natively in one instruction stream.

**Expected impact:** small (~5%); only relevant after #1 lands.

---

## 6. Built-in NB ping-pong on `pto.alloc_tile`

**Symptom:** the natural Python pattern `bufs = [pto.alloc_tile(ty)] * 2`
**crashes ptoas** during LLVM lowering (same SSA value used as multiple ops'
output). The required workaround is `[pto.alloc_tile(ty), pto.alloc_tile(ty)]`
(two distinct allocs aliased into a list) — non-obvious footgun.

**Need:** add `nb: int = 1` kwarg to `pto.alloc_tile` returning a list of N
distinct allocations.

**Effort:** ~5 lines + a tiny convention. Trivial; UX/footgun fix.

---

## 7. Reduction ops on subviews

**Status:** automatic once #1 lands. The vec ops (`tile.row_max`, `row_sum`,
`row_expand_*`) already accept any TileBuf with a strided layout — they just
have no way to receive a subview today.

---

## 8. Causal mask

**Need:** either a fused `tile.causal_mask_softmax(qk, q_offset, k_offset)`
op, or expose primitives `tile.row_index` + `tile.cmpgt` + `tile.select` so
users can build it. No DSL primitive currently materializes a triangular mask.

**Status:** blocks porting to typical LLM workloads (decode/prefill with KV
cache); not on the critical path for the current non-causal benchmark.

---

## Summary

| # | Feature | Blocker type | Effort | Impact |
|---|---|---|---|---|
| **1** | **Sub-tile views on TileBuf** | API + matmul overload | medium | **2× vec amortization** |
| 2 | Batched consumer-acks | API plumbing | trivial | 15–25% |
| 3 | Decoupled `local_slot_num` (legacy pipes) | API plumbing | trivial | enables larger tiles |
| 4 | PRELOAD≥4 (event pool) | constant bump OR new pass logic | trivial→hard | 10–15% |
| 5 | Multi-K fused matmul | New op | medium | ~5% |
| 6 | NB ping-pong on `alloc_tile` | API ergonomics | trivial | UX (fixes a crash) |
| 7 | Subview reductions | downstream of #1 | free | small |
| 8 | Causal mask | New op family | medium | unblocks LLM workloads |

**Quick wins for the compiler team:** #2, #3, #6, and #4-option-(c) are all
either pure plumbing or one-line constant changes. Landing those four alone
should noticeably improve perf and remove the worst footguns. **#1 is the
single biggest perf win** but needs design discussion on how subviews should
be spelled (op shape, stride ownership, lowering).
