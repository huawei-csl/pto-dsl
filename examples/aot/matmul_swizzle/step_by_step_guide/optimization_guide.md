# PTO DSL Matmul Optimization Guide (4 Steps)

This tutorial walks through a practical optimization path for dynamic-shape matmul on NPU using the ptodsl framework.

The key idea: **keep correctness fixed**, then change only one optimization dimension at a time so each speedup is easy to understand and measure.

---

## 1) Mental Model: What each step changes

- **Step1 (`step1_baseline.py`)**: functionally correct baseline; simple tile order; single L1 buffers.
- **Step2 (`step2_doublebuffer.py`)**: add double-buffering for A/B tiles (overlap data movement with compute), still linear tile order.
- **Step3 (`step3_swizzle.py`)**: keep double-buffering and add swizzled tile traversal to improve access/balance patterns.
- **Step4 (`step4_manual_pipelining.py`)**: keep step3 algorithm but replace compiler auto-sync with explicit event-driven software pipeline.

---

## 2) Shared Building Blocks (`common_utils.py`)

All steps reuse the same tile sizes, metadata, and swizzle helper.

### Why shared utilities matter

- Keeps step diffs focused on optimization logic.
- Reduces accidental config drift across kernels.
- Makes benchmarking comparisons fair.

### Key shared code

```python
M_TILE = 128
K_QTILE = 64
K_TILE = 256
K_DTILE = 512
N_FULL = 256
SWIZZLE_COUNT = 5
```

```python
def build_meta_data():
    def meta_data():
        dtype = pto.float16
        acc_dtype = pto.float32
        ptr_type = pto.PtrType(dtype)
        i32 = pto.int32
        tv_2d = pto.TensorType(rank=2, dtype=dtype)
        ...
```

```python
def swizzle_nz(li, m_loop, n_loop, c_swizzle, c_swizzle_m1, c1, c2):
    tile_block_loop = (n_loop + c_swizzle_m1) // c_swizzle
    tile_block_span = c_swizzle * m_loop
    tile_block_idx = li // tile_block_span
    ...
    m_idx = s.select(odd_block, flipped_m_idx, m_idx)
    return m_idx, n_idx
```

If `swizzle_nz` looks confusing: think of it as remapping linear tile index `li` into a 2D `(m_idx, n_idx)` traversal order that improves behavior compared with pure row-major tile walking.

---

## 3) Step1 Baseline: Correctness-first kernel

File: `step1_baseline.py`

### Algorithm behavior

- Dynamic shape support from runtime `(m, n, k)` parameters.
- Tiles are visited in plain linear order:
  - `m_idx = li // n_loop`
  - `n_idx = li % n_loop`
- One L1 tile for A and one L1 tile for B (no ping-pong buffers).
- No explicit pipeline/event synchronization.

### Important code

```python
for li in pto.range(bid, core_loop, num_blocks):
    m_idx = li // n_loop
    n_idx = li % n_loop
    m_offset = m_idx * c128
    n_offset = n_idx * c256
```

```python
a_l1 = pto.alloc_tile(tile_buf_a_l1)
b_l1 = pto.alloc_tile(tile_buf_b_l1_256)
...
pto.load(sv_a0, a_l1)
...
pto.load(sv_b, b_l1)
```

```python
if phase == 0:
    with pto.if_context(is_first_k_tile, has_else=True) as branch:
        tile.matmul(a_l0, b_l0, c_l0)
    with branch.else_context():
        tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)
else:
    tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)
```

### Why this is the baseline

It is easy to reason about and debug. Every later step should preserve this numerical result.

### NumPy simulation of Step1 (algorithm teaching version)

The full code is in `step1_numpy_sim.py`.

Run it directly:

```bash
python ./step1_numpy_sim.py
```

### Line-by-line mapping to `step1_baseline.py`

- **Loop space construction**
  - NumPy: `n_loop`, `m_loop`, `core_loop`, `k_dtile_num`
  - ptodsl: same scalar setup in `step1_baseline.py`
- **Core tile traversal**
  - NumPy: `for li in range(core_loop)`
  - ptodsl: `for li in pto.range(bid, core_loop, num_blocks)`
- **Tile index mapping**
  - NumPy: `m_idx = li // n_loop`, `n_idx = li % n_loop`
  - ptodsl: same formulas
- **K loop**
  - NumPy: `for k_idx in range(k_dtile_num)`
  - ptodsl: `for k_idx in pto.range(c0, k_dtile_num, c1)`
- **Phase loop (build-time unrolled in ptodsl)**
  - NumPy: `for phase in range(8)`
  - ptodsl: same Python loop, used for static unrolling in IR build
- **First-accumulate logic**
  - NumPy: `if phase == 0 and is_first_k_tile: c_tile = prod else: c_tile += prod`
  - ptodsl: `if phase == 0` + `pto.if_context(is_first_k_tile, has_else=True)` with `matmul` / `matmul_acc`

### Why `b_l0.T` is needed (and how it maps to ptodsl)

In this tutorial, `b` is stored as shape `[n, k]`, while `a` is `[m, k]`.

- NumPy quarter tile:
  - `a_l0` shape is `[M_TILE, K_QTILE]`
  - `b_l0` shape is `[N_FULL, K_QTILE]`
- To compute output tile `[M_TILE, N_FULL]`, we need:
  - `[M_TILE, K_QTILE] @ [K_QTILE, N_FULL]`
  - therefore NumPy uses `a_l0 @ b_l0.T`

In ptodsl, this transpose handling is embedded by the tensor/view layout settings and tile ops:
- `tv_b` is created with `layout="DN"` in `step1_baseline.py`
- `tile.extract(...)` and `tile.matmul(...)` then consume B in the expected orientation for GEMM

So `b_l0.T` in NumPy is the explicit equivalent of what ptodsl layout + tile pipeline already encodes implicitly.

### Why accumulate in `float32`

The original kernel metadata sets:
- input dtype: `float16`
- accumulator dtype: `float32`

That is why the NumPy simulation casts tile loads to `float32` and keeps `c_tile`/`c` as `float32`. This mirrors:
- `acc_dtype = pto.float32`
- `tile_buf_c_256` using `acc_dtype`

Using float32 accumulation is important for numerical stability across many partial products (especially large K).

---

## 4) Step2 Double-buffer: overlap movement and compute

File: `step2_doublebuffer.py`

### Algorithm delta from Step1

- Change single buffers into ping-pong buffers:
  - `a_l1 = [buf0, buf1]`
  - `b_l1 = [buf0, buf1]`
- Keep tile traversal **non-swizzled** (same simple `m_idx/n_idx` as baseline).
- Keep autosync flow (no explicit manual event schedule in source).

### Important code

```python
a_l1 = [pto.alloc_tile(tile_buf_a_l1), pto.alloc_tile(tile_buf_a_l1)]
b_l1 = [pto.alloc_tile(tile_buf_b_l1_256), pto.alloc_tile(tile_buf_b_l1_256)]
...
is_curr0 = (k_idx % c2) == c0
with pto.if_context(is_curr0, has_else=True) as branch:
    run_loop_k(a_l1[0], a_l1[1])
with branch.else_context():
    run_loop_k(a_l1[1], a_l1[0])
```

```python
def run_loop_k(a_curr, a_next):
    ...
    tile.extract(a_curr, c0, a_col, a_l0[ping])
    ...
    with pto.if_context(k_idx + c1 < k_dtile_num):
        ...
        pto.load(sv_a_next, a_next)
```

### Why this tends to speed up

While compute is consuming `a_curr`, the next tile can be prepared into `a_next`, reducing pipeline bubbles.

Tiny timeline sketch (conceptual):

```text
Step2 (double-buffer, auto-sync)
time ---->
Load A/B buf0: [====]
Compute  buf0:       [========]
Load A/B buf1:           [====]
Compute  buf1:                 [========]
```

---

## 5) Step3 Swizzle: improve tile traversal pattern

File: `step3_swizzle.py`

### Algorithm delta from Step2

- Keep same double-buffer kernel structure.
- Only change the mapping from linear loop index `li` to tile coordinates `(m_idx, n_idx)`:
  - from linear mapping
  - to `swizzle_nz(...)` mapping

### Important code

```python
c_swizzle = const(SWIZZLE_COUNT)
c_swizzle_m1 = c_swizzle - c1
...
m_idx, n_idx = swizzle_nz(li, m_loop, n_loop, c_swizzle, c_swizzle_m1, c1, c2)
```

Everything else (double-buffer loop body) stays essentially the same as Step2, which makes Step2 -> Step3 comparison clean.

### Intuition for new users

Swizzling is **not changing math**, only **work scheduling order**. On NPUs, scheduling order can strongly affect memory traffic and utilization.

### NumPy swizzle mapping demo

To make swizzle behavior concrete, use `step3_swizzle_numpy_sim.py`.
It prints tile index mapping before/after swizzle for several swizzle factors.

```bash
python ./step3_swizzle_numpy_sim.py
```

Example output format:

```text
=== swizzle=5, m_loop=4, n_loop=7, core_loop=28 ===
li | linear(m,n) -> swizzle(m,n)
 0 | ( 0, 0) -> ( 0, 0)
 1 | ( 0, 1) -> ( 0, 1)
 2 | ( 0, 2) -> ( 0, 2)
 ...
```

Interpretation:
- `linear(m,n)` is the baseline order (`m_idx = li // n_loop`, `n_idx = li % n_loop`).
- `swizzle(m,n)` is the remapped order used by `swizzle_nz(...)`.
- As you vary `c_swizzle` (2, 3, 5), you can see how traversal shape and direction change, especially near N-tail blocks.
- The script also prints 2D order grids:
  - `linear_order_grid[m, n] = li` in baseline traversal
  - `swizzle_order_grid[m, n] = li` in swizzled traversal
  This gives an intuitive “heatmap-like” view of where each tile is visited in time.

---

## 6) Step4 Manual Pipelining: explicit software schedule

File: `step4_manual_pipelining.py`

### Algorithm delta from Step3

- Keep swizzled traversal and double-buffer dataflow.
- Switch from autosync-style source to explicit event orchestration:
  - `record_event(...)`
  - `wait_event(...)`
  - `record_wait_pair(...)`

### Important code

```python
pto.record_event("MATMUL", "MOV_M2L", event_id=[0, 1])
pto.record_event("MOV_M2L", "LOAD", event_id=[0, 1, 2, 3])
```

```python
pto.wait_event("MOV_M2L", "LOAD", event_id=b_evt)
pto.load(sv_b, b_l1[h])
pto.record_event("LOAD", "MOV_M2L", event_id=b_evt)
```

```python
pto.wait_event("MOV_M2L", "MATMUL", event_id=0)
...
pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
pto.store(c_l0, sv_c)
```

### Why this can help

Manual scheduling gives tighter control over producer-consumer ordering and overlap. It often improves tail behavior and removes conservative compiler sync points.

Tiny timeline sketch (conceptual):

```text
Step4 (manual pipeline with explicit events)
time ---->
LOAD ----record----> MOV_M2L ----record----> MATMUL ----record----> STORE
   ^                      |                        |                    |
   |------ wait ----------+------ wait -----------+------ wait --------+
```

---

## 7) Build and Run

### Build all 4 steps

```bash
bash ./compile.sh
```

Artifacts are generated in `build_artifacts/`:
- `step1_baseline_kernel.so`
- `step2_doublebuffer_kernel.so`
- `step3_swizzle_kernel.so`
- `step4_manual_pipelining_kernel.so`

### Validate correctness

```bash
python ./run_simple_matmul.py
```

Run one step only:

```bash
python ./run_simple_matmul.py --variant step1-baseline
python ./run_simple_matmul.py --variant step2-doublebuffer
python ./run_simple_matmul.py --variant step3-swizzle
python ./run_simple_matmul.py --variant step4-manual-pipelining
```

### Run stepwise benchmark

```bash
python ./bench_matmul.py
```

---

## 8) Interpreting benchmark ratios

The benchmark prints three ratio groups:

1. **Step1 ratio**: `step2 / step1`
   - isolates gain from double-buffering.
2. **Step2 ratio**: `step3 / step2`
   - isolates gain from swizzle.
3. **Step3 ratio**: `step4 / step3`
   - isolates gain from manual software pipelining.

Reference result:

```text
=== Summary ===
Step1 (double-buffer speedup, both non-swizzle auto-sync):
avg FLOP ratio(double_noswizzle_auto/single_noswizzle): 1.607x
min FLOP ratio(double_noswizzle_auto/single_noswizzle): 0.943x
max FLOP ratio(double_noswizzle_auto/single_noswizzle): 1.826x
Step2 (swizzle speedup, both double-buffer auto-sync):
avg FLOP ratio(double_swizzle_auto/double_noswizzle_auto): 1.227x
min FLOP ratio(double_swizzle_auto/double_noswizzle_auto): 0.863x
max FLOP ratio(double_swizzle_auto/double_noswizzle_auto): 1.871x
Step3 (manual-sync speedup, both double-buffer swizzle):
avg FLOP ratio(double_swizzle_manual/double_swizzle_auto): 1.100x
min FLOP ratio(double_swizzle_manual/double_swizzle_auto): 1.001x
max FLOP ratio(double_swizzle_manual/double_swizzle_auto): 1.173x
```

---

## 9) Suggested learning path

- First, run `step1` only and inspect correctness outputs.
- Next, compare `step1` vs `step2` source side by side, focusing on buffer allocation and `run_loop_k`.
- Then inspect only the index mapping change from `step2` to `step3`.
- Finally, study `step4` event dependencies as a timeline (LOAD -> MOV_M2L -> MATMUL -> STORE).

If you keep this one-change-per-step mindset, it becomes much easier to learn NPU kernel optimization systematically.

---

## Appendix A) ptodsl Syntax for Python Users

If you are new to ptodsl, the biggest source of confusion is:
- some syntax is **Python control flow**
- some syntax is **IR-builder control flow**

They look similar, but they execute at different times.

### Build-time vs run-time cheat sheet

- **Python `for ... in range(...)`**
  - runs when generating the IR (build-time)
  - usually acts like compile-time metaprogramming/unrolling
- **`for ... in pto.range(...)`**
  - emits an MLIR `scf.for` loop
  - executes dynamically at kernel run-time
- **Python `if condition:`**
  - condition evaluated at build-time by Python
  - branch is selected while generating IR
- **`with pto.if_context(cond):` / `pto.cond(...)`**
  - emits runtime `scf.if`
  - condition is evaluated when kernel runs

### Example 1: `pto.range` (runtime loop in IR)

From `step1_baseline.py`:

```python
for li in pto.range(bid, core_loop, num_blocks):
    ...
```

This is **not** Python iteration over integers. In ptodsl, `pto.range` is an IR-builder primitive (see `control_flow.py`) that constructs `scf.ForOp` and yields an induction variable value.

Practical effect:
- loop trip count depends on runtime values like `bid`, `core_loop`, `num_blocks`
- loop stays as a loop in generated IR (not unrolled by Python)

### Example 2: Python `range` (build-time unrolling)

From `step1_baseline.py`:

```python
for phase in range(8):
    ...
```

This loop is executed by Python while building IR, so it typically creates 8 repeated code regions in IR.

For readers with C++ background:
- this is conceptually similar to compile-time code generation / metaprogramming
- useful when loop bounds are small constants

### Example 3: Python `if` vs `pto.if_context`

From `step1_baseline.py`:

```python
if phase == 0:
    with pto.if_context(is_first_k_tile, has_else=True) as branch:
        tile.matmul(a_l0, b_l0, c_l0)
    with branch.else_context():
        tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)
else:
    tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)
```

How to read this correctly:
- `if phase == 0` is a **Python** branch (build-time), because `phase` is a Python integer from `range(8)`.
- `pto.if_context(is_first_k_tile, ...)` emits a **runtime** branch in IR, because `is_first_k_tile` is a kernel scalar value.

In plain words:
- first, Python decides which code shape to generate for each unrolled `phase`
- inside that shape, ptodsl inserts dynamic control flow for runtime conditions

### Rule of thumb

When in doubt, ask:
1. Is this condition/index a Python value (`int`, `bool`)?
   - then it is build-time.
2. Is this a ptodsl scalar/value (`s.*`, kernel arg-derived)?
   - then use ptodsl control flow (`pto.range`, `pto.if_context`, `pto.cond`) for runtime behavior.
