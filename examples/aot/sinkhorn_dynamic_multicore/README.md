# Sinkhorn normalization (dynamic-batch, multicore)

PTODSL implementation of the Sinkhorn-style row/column normalization kernel
defined in [reference.cpp](reference.cpp). The kernel iteratively rescales
two diagonal vectors `mu1[L]` and `mu2[K]` so that the row and column
standard deviations of `matrix_in / (mu2[:, None] * mu1[None, :])` converge
to a common target value.

## Algorithm

For each `(K, L)` matrix in a batch of `N`:

1. Initialise `mu1 = mu2 = invMu1 = 1`.
2. For `phase = 0 .. order`:
   - Compute row & column **unbiased** standard deviations of
     `cm = matrix_in / (mu2 * mu1)` in chunks of `ROW_CHUNK = 8` rows.
   - `phase == 0`: set `tgt = min(rStd.min(), cStd.min()) + eps`.
   - `phase > 0` : `mu2 *= (rStd / tgt) ** lr` and `mu1 *= (cStd / tgt) ** lr`,
     then refresh `invMu1 = 1 / mu1`.
3. Write `matrix_out = matrix_in / (mu2 * mu1)`, plus `mu1_out`, `mu2_out`.

## Design choices vs the hand-tuned reference

The reference C++ exists primarily to squeeze every last cycle out of the
hardware (templated `TileL`, manual UB layout, hand-pipelined `set_flag` /
`wait_flag`, 2-term Padé `approxLn`). The PTODSL version trades a small
amount of throughput for clarity:

| Concern               | Reference                          | PTODSL builder                           |
| --------------------- | ---------------------------------- | ---------------------------------------- |
| Per-`L` specialisation| `runSinkhornImpl<T, TileL>` switch | Single `MAX_DIM = 256` column stride     |
| `inv_mu1` broadcast   | Pre-tiled to `[ROW_CHUNK, L]` buf  | `tile.col_expand_mul`                    |
| `pow(x, lr)`          | 2-term Padé `approxLn` + `TEXP`    | Native `tile.log` / `tile.exp`           |
| Pipe synchronisation  | Manual `set_flag` / `wait_flag`    | `ptoas --enable-insert-sync`             |

## Constraints

- `1 <= K, L <= 256` (`MAX_DIM`).
- `K % 8 == 0` (`ROW_CHUNK`). Tail handling for non-aligned `K` is left to
  a future revision; the reference handles it via dynamic `cr`.
- Inputs are `fp16`; internal compute is `fp32`.

## PTODSL / PTOAS workarounds

Two limitations of the current stack forced extra plumbing compared to the
reference. Both are pure boilerplate and could be removed by toolchain
fixes.

| # | Workaround in [sinkhorn_builder.py](sinkhorn_builder.py) | Root cause | Suggested fix |
| - | -------------------------------------------------------- | ---------- | ------------- |
| 1 | `mu2` is held as `RowMajor [1, MAX_DIM]` then per-chunk **copied** into a static `[1, ROW_CHUNK]` tile (`mu2RowStatic`) before being reshaped to the col-major `[ROW_CHUNK, 1]` sibling fed to `tile.row_expand_div`. | `pto.subview` narrows the *valid* shape but reuses the parent's *storage* `Numel`, so a downstream `tile.reshape` fails the bisheng `TRESHAPE` byte-size `static_assert` (parent `Numel = 256` ≠ `8`). | Have `pto.subview` rewrite the result tile-buf type's storage `shape` to the slice sizes when those sizes are static, so subview→reshape round-trips through a tile whose `Numel` matches what TRESHAPE expects. Alternatively expose a typed view-cast op (the new `pto.bitcast`/`pto.set_validshape` cover dtype/valid-shape but not storage-shape narrowing). |
| 2 | A static `[1, 1]` "scalar" tile is allocated as `[8, 1]`/`[1, 8]` with dynamic `valid_shape=[-1, -1]` so that `tile.min` / `tile.adds` / `tile.reshape` find the runtime `GetValidRow/Col` they require even though the value is conceptually 1×1. | The verifier+codegen for `TMin`/`TAddS`/`TRESHAPE` requires dynamic-valid metadata even on degenerate 1×1 tiles, and there's no row-major scalar type accepted by `tile.row_expand_div` as a broadcast source. | Lower `TMin`/`TAddS` over a fully-static `1×1` tile by emitting the immediate-form intrinsic directly, and let `tile.row_expand_div` accept a row-major `[1, 1]` scalar source (broadcast over both axes). |

A third minor item: every K-indexed quantity (`mu2`, `rowSum`, `rowSqsum`)
is forced into `RowMajor [1, MAX_DIM]` instead of the natural `ColMajor
[MAX_DIM, 1]` because none of the elementwise ops
(`TMul/TSub/TMin/TLog/TExp/TSqrt/TAddS/TRowMin/T*ExpandDiv`) accept a
layout-override attribute. Adding such an attribute would let the builder
keep K-vectors col-major and drop the per-chunk reshape entirely.

## Test coverage

[run_sinkhorn.py](run_sinkhorn.py) runs the same matrix the upstream
torch_npu suite uses: `11 shapes × {order ∈ {1, 5, 10}} × {seed ∈ {0,
42}} = 66 cases`, including non-square `(1, 16, 256)` and `(1, 256, 16)`,
batched `(8, 128, 128)`, and the boundary `(1, 256, 256)`. Tolerances
match upstream (`rtol=5e-2`, `atol=1e-2`). All 66 cases pass against the
PyTorch reference.

## Files

| File                    | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `sinkhorn_builder.py`   | PTODSL kernel — emits MLIR via stdout                      |
| `caller.cpp`            | Thin C wrapper, exports `call_sinkhorn_kernel`             |
| `compile.sh`            | `python builder > .pto` → `ptoas` → `bisheng` shared lib   |
| `run_sinkhorn.py`       | Numerical correctness vs PyTorch reference                 |
| `reference.cpp`         | Hand-tuned baseline (`call_sinkhorn_kernel` self-contained)|
| `jit_util_sinkhorn.py`  | Cached JIT compile + `ctypes` loader for both kernels      |
| `bench_sinkhorn.py`     | Throughput benchmark (torch / PTODSL / reference)          |

## Usage

```bash
# 1. Generate MLIR + compile shared library (inside the NPU container).
./compile.sh

# 2. Run correctness check (66 cases mirroring upstream torch_npu suite).
python ./run_sinkhorn.py --lib ./sinkhorn_lib.so

# 3. JIT-compile both kernels and benchmark.
python ./bench_sinkhorn.py
# Outputs:
#   outputs/csv/{head_shapes_bench,batched_vs_serial}.csv
#   outputs/plots/head_shapes_*.png, batched_vs_serial_log.png
```

## Throughput (Atlas 800I A2, fp16, order=8, lr=0.9, eps=1e-6)

Single-matrix latency over the transformer-head grid (K ∈ {64, 128, 256},
L ∈ {32, 64, 128, 256}), 5 warmup + 20 timed runs:

| K   | L   | torch fp16 (µs) | PTODSL (µs) | reference C++ (µs) | PTODSL / torch | PTODSL / ref |
| --: | --: | --------------: | ----------: | -----------------: | -------------: | -----------: |
| 64  | 32  | 2170            | 39          | 65                 | **55.6×**      | **1.66×**    |
| 64  | 256 | 2017            | 57          | 81                 | 35.1×          | 1.41×        |
| 128 | 128 | 1983            | 79          | 124                | 25.1×          | 1.56×        |
| 256 | 256 | 2012            | 200         | 282                | 10.1×          | 1.41×        |

Across all 12 shapes the PTODSL kernel is **10–55× faster than torch
fp16** and **1.40–1.78× faster than the hand-tuned reference C++**. The
batched-vs-serial sweep (K = L = 128) shows PTODSL holds the same ~80 µs
from N = 1 to N = 32 (perfect multicore scaling), consistently
**~1.55–1.63×** ahead of the reference at every batch size.
