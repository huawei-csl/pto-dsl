# Sinkhorn K=4: from PyTorch reference to PTO tile ops

This note is for readers who know PyTorch but are new to **PTO vector programming**. It walks through how the numeric recipe in `sinkhorn_normalize_ref` appears in `sinkhorn_k4_builder.py`, and how familiar PyTorch calls relate to **tile** primitives (which lower to PTO-ISA instructions such as `TROWSUM`, `TCOLSUM`, and friends).

---

## 1. What the PyTorch reference is doing

Consider the last two dimensions as an **K×K** matrix (here K = 4). In PyTorch, the **last** dimension (`-1`) is the **column** index, and the **second-to-last** (`-2`) is the **row** index.

```python
def sinkhorn_normalize_ref(x, repeat=10, eps=1e-6):
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x
```

| Step | PyTorch | Meaning on the K×K block |
|------|---------|----------------------------|
| A | `softmax(-1)` | For **each row**, turn logits into a probability vector (non-negative, sum to 1 along **columns**). |
| B | `+ eps` | Add a small constant everywhere (numerical stability). |
| C | `x / (x.sum(-2, ...) + eps)` | **Column** normalization: for each **column**, divide every entry in that column by the column sum (+ ε). |
| Loop | `sum(-1)` then `sum(-2)` | **Row** normalize, then **column** normalize, repeated `repeat - 1` times. |

So the structure is: **softmax along rows → +ε → one column-normalize → (row-normalize → column-normalize) × (repeat−1)**.

---

## 2. Mental model: rows vs columns in reductions

- **`sum(-1)`** (sum over **last** axis) = sum **across columns** within each row → one number **per row** → in PTO you see **`row_sum`**, **`row_max`**, etc., with a **column-shaped** “stat” tile (one slot per row).
- **`sum(-2)`** (sum over **second-last** axis) = sum **across rows** within each column → one number **per column** → in PTO you see **`col_sum`**, with a **row-shaped** “stat” tile (one slot per column).

**Broadcasting** in PyTorch (`keepdim=True`) becomes **expand** ops in PTO: a per-row statistic is **replicated along columns** (`row_expand_*`); a per-column statistic is **replicated along rows** (`col_expand_*`).

---

## 3. Line-by-line mapping (builder core vs reference)

Inside the `pto.range` loop over matrices, `mat_kk` is the working **K×K** tile (the same mathematical object as one K×K slice of `x`). The lines below map to the reference in order.

### Setup (not in the PyTorch one-liners, but required on device)

| Builder | Role |
|---------|------|
| `tile.muls(mat_full, f0_h, mat_full)` | Clear the padded UB tile (hardware uses a larger 16×16 buffer; padding must not affect reductions). |
| `pto.load(gm_in, mat_kk)` | Copy this batch element’s K×K matrix from **global memory (GM)** into the tile. |

### Phase A–B: `x = x.softmax(-1) + eps`

PyTorch does **stabilized softmax**: subtract row max, exponentiate, divide by row sum of exponentials, then add ε.

| PyTorch idea | PTO-DSL / `tile` API | ISA-style intuition |
|--------------|----------------------|---------------------|
| Row-wise max for softmax stability | `tile.row_max(mat_kk, scratch_kk, row_stat)` | One max **per row** into `row_stat` |
| Subtract that max from each row | `tile.row_expand_sub(mat_kk, row_stat, mat_kk)` | Broadcast row stats along columns, subtract |
| `exp` | `tile.exp(mat_kk, mat_kk)` | Elementwise |
| Sum of exponentials per row | `tile.row_sum(mat_kk, scratch_kk, row_stat)` | One sum **per row** |
| Divide each row by its sum | `tile.row_expand_div(mat_kk, row_stat, mat_kk)` | Softmax along `-1` is complete |
| `+ eps` | `tile.adds(mat_eps_rows, eps_h, mat_eps_rows)` | Add ε on the first **K×16** region (matches the hand kernel’s padding/stride pattern; mathematically the extra positions are zero before this, so the K×K block behaves like `+ eps` on the matrix). |

**Beginner tip:** reductions (`row_max`, `row_sum`, `col_sum`) often need a **scratch** tile (`scratch_kk`) as temporary storage; always check the op’s signature in `ptodsl.api.tile`.

### Phase C: first column normalize — `x / (x.sum(-2, ...) + eps)`

| PyTorch | PTO-DSL | Comment |
|---------|---------|--------|
| `x.sum(-2, keepdim=True)` | `tile.col_sum(mat_kk, scratch_kk, col_stat, is_binary=True)` | One sum **per column** into `col_stat` |
| `+ eps` | `tile.adds(col_stat, eps_h, col_stat)` | ε on each column statistic |
| Division broadcast on columns | `tile.col_expand_div(mat_kk, col_stat, mat_kk)` | Each column divided by its (sum + ε) |

### Loop: `for _ in range(repeat - 1):` row then column

`pto.range(c1, repeat_idx, c1)` is the **runtime** loop over the iteration count (like Python `for _ in range(repeat - 1)` when tracing the IR). Each iteration does:

| PyTorch (one loop iteration) | PTO-DSL |
|--------------------------------|---------|
| `x / (x.sum(-1, ...) + eps)` | `row_sum` → `adds` on `row_stat` → `row_expand_div` |
| `x / (x.sum(-2, ...) + eps)` | `col_sum` → `adds` on `col_stat` → `col_expand_div` |

So the **body** of `pto.range(c1, repeat_idx, c1)` matches the **body** of the PyTorch `for` loop exactly: **row normalize, then column normalize**.

### Writeback

| Builder | PyTorch analogy |
|---------|-----------------|
| `pto.store(mat_kk, gm_out)` | Writing the final K×K tensor back to the output tensor in memory. |

---

## 4. Quick reference table: PyTorch ↔ PTO (`tile` helpers)

| PyTorch (on last K×K block) | Typical PTO pattern |
|-----------------------------|---------------------|
| `softmax(-1)` | `row_max` → `row_expand_sub` → `exp` → `row_sum` → `row_expand_div` |
| `x + scalar` | `tile.adds(...)` (or add tiles as supported) |
| `x / (x.sum(-1, keepdim=True) + eps)` | `row_sum` → `adds` on stat → `row_expand_div` |
| `x / (x.sum(-2, keepdim=True) + eps)` | `col_sum` → `adds` on stat → `col_expand_div` |

---

## 5. PTO concepts worth internalizing early

1. **GM vs UB:** PyTorch tensors live in **global memory**; the kernel does math on **on-chip (UB) tiles**. You **load** once, **compute** many steps, **store** once (`pto.load` / `pto.store`).
2. **Row vs column ops:** Names are from the **statistic** layout: **row_*** produces one value per **row** (summed/max over columns). **col_*** produces one value per **column**.
3. **Expand:** PyTorch broadcasting of a row or column vector to a matrix is explicit: **`row_expand_*`** / **`col_expand_*`**.
4. **Runtime vs build-time loops:** In PTO-DSL, **`pto.range`** (and `pto.if_context`, etc.) becomes **real** control flow on the device. A plain Python `for` over a constant range is often **unrolled at trace time**; here the iteration count is **dynamic** (`repeat`), so **`pto.range`** is the right tool (see the skill doc on “runtime vs build-time” in the translate-cpp2py skill).
5. **Padding (K=4, tile 16×16):** Vector hardware often requires a minimum tile width; the builder uses a 16×16 UB tile but only **K×K** math matters for correctness if padding is zeroed and reductions respect valid regions as in this kernel.

---

## 6. Flow diagram (same order as code)

```text
GM [K×K]  --load-->  mat_kk
                         |
            +----------+----------+
            | softmax(-1)        |
            v                    |
    row_max -> row_expand_sub -> exp -> row_sum -> row_expand_div
            |                    |
            +--------+-----------+
                     |
              adds eps (mat_eps_rows)
                     |
              col_sum -> adds(eps) -> col_expand_div   ← first col norm (ref line after softmax)
                     |
         +-----------+-----------+
         | pto.range (repeat-1)  |
         v                       |
    row_sum -> adds -> row_expand_div    ← sum(-1) normalize
         |                       |
    col_sum -> adds -> col_expand_div    ← sum(-2) normalize
         |                       |
         +-----------------------+
                     |
              store --> GM [K×K]
```

With this picture, you can read **`sinkhorn_k4_builder.py`** side by side with **`sinkhorn_normalize_ref`** and see the same alternating **row / column** normalization pattern, with softmax decomposed into the usual stable **row-wise** building blocks.
