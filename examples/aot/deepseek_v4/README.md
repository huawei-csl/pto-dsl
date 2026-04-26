# DeepSeek-V4 PTO ports

PTO DSL ports of the five custom kernels used by the DeepSeek-V4 reference
implementation. Every kernel is self-contained in its own folder and follows
the same `compile.sh` → `pytest` → optional `bench_*.py` workflow.

| Folder                              | What it does                                                                          | Pipe(s)         |
|-------------------------------------|---------------------------------------------------------------------------------------|-----------------|
| [act_quant/](act_quant/)            | Per-row absmax fp16 → int8 quant (`max(|x|)/127`, `round(x/scale)`)                   | vector          |
| [fp4_act_quant/](fp4_act_quant/)    | Per-row fp16 → mxfp4 (e2m1) quant with shared exponent + lookup-table cast            | vector          |
| [fp8_gemm/](fp8_gemm/)              | Per-channel fp8 (e4m3) GEMM with host-side fused `Sa`/`Sb` pre-scale                  | cube + vector   |
| [fp4_gemm/](fp4_gemm/)              | Per-channel fp4 (e2m1) GEMM with host-side fused `Sa`/`Sb` pre-scale                  | cube + vector   |
| [hc_split_sinkhorn/](hc_split_sinkhorn/) | Fused MoE-router head: pre/post sigmoid + 20-iter Sinkhorn, all on-device         | vector          |
| [sparse_attn/](sparse_attn/)        | FlashAttention with indexed top-k KV gather + per-head sink logit                     | vector          |

> ✅ **34 tests, 0 xfailed** across the full suite (last verified: this branch).

---

## Build a single kernel

Each folder ships a `compile.sh` that:

1. runs `python *_builder.py` to emit a `.pto` file,
2. runs `ptoas --enable-insert-sync` to lower to a `.cpp`,
3. runs `bisheng` to produce a `*_lib.so`.

```bash
cd examples/aot/deepseek_v4/sparse_attn
bash compile.sh
```

The generated `.pto`, `.cpp` and `.so` files are gitignored.

## Build everything

```bash
for d in examples/aot/deepseek_v4/*/; do
    [[ -f "$d/compile.sh" ]] && (cd "$d" && bash compile.sh) || true
done
```

## Run all tests

From the repo root, after the kernels are built:

```bash
python -m pytest examples/aot/deepseek_v4/ -v
```

Expected: **34 passed, 0 xfailed** in ~3 s.

To run a single kernel's tests:

```bash
python -m pytest examples/aot/deepseek_v4/sparse_attn/ -v
```

> Tests skip with a clear message if the corresponding `*_lib.so` is missing.

## Run benchmarks

Two kernels currently ship microbenchmarks; the rest only have correctness
tests (their reference paths are trivially short and not interesting to time
in isolation).

### sparse_attn

Compares the PTO kernel against:
- the eager PyTorch reference (slow, exact, including the sink logit);
- a *realistic* PyTorch-on-NPU baseline: `torch.gather` of the K KV rows
  followed by `torch_npu.npu_fused_infer_attention_score` in MQA mode
  (`num_key_value_heads=1`) — sink logit is dropped, so this is a *speed*
  baseline only.

```bash
cd examples/aot/deepseek_v4/sparse_attn
bash compile.sh
python bench_sparse_attn.py
```

Sample output (this branch, single-card warm cache):

```
  B   M     N    K     pto us     ref us   fused us   pto/ref  pto/fused
------------------------------------------------------------------------
  1   1   128   64     161.15     533.05     265.03     3.31x      1.64x
  1   4   256  128     209.56    1692.93     252.36     8.08x      1.20x
  2   2   512  128     208.87    1628.79     255.80     7.80x      1.22x
  4   4  1024  128     207.77    6071.60     246.57    29.22x      1.19x
  8   8  2048  128     304.49   24658.49     244.67    80.98x      0.80x
```

### hc_split_sinkhorn

Compares the fused on-device kernel against the eager PyTorch reference
(pre/post sigmoid heads + 20-iter Sinkhorn).

```bash
cd examples/aot/deepseek_v4/hc_split_sinkhorn
bash compile.sh
python bench_hc_split_sinkhorn.py
```

Sample output:

```
      n     pto us     ref us  speedup
----------------------------------------
     64     173.27    2803.42   16.18x
    256     151.08    2777.09   18.38x
   1024     218.70    2761.33   12.63x
   4096     532.02    2761.56    5.19x
  16384    1786.32    2741.09    1.53x
```

## End-to-end (build → test → bench)

```bash
# 1. Build all five kernels
for d in examples/aot/deepseek_v4/*/; do
    [[ -f "$d/compile.sh" ]] && (cd "$d" && bash compile.sh) || true
done

# 2. Validate correctness
python -m pytest examples/aot/deepseek_v4/ -v

# 3. Time the two kernels with benchmarks
python examples/aot/deepseek_v4/sparse_attn/bench_sparse_attn.py
python examples/aot/deepseek_v4/hc_split_sinkhorn/bench_hc_split_sinkhorn.py
```

## Implementation notes

- **`fp8_gemm` / `fp4_gemm`** — the GPU op fuses an outer `Sa[m] * Sb[n]`
  per-channel rescale into the GEMM. The PTO kernels keep the matmul
  pure (cube fp32 accum → fp16 cast) and instead **pre-scale `A` on the
  host** by `Sa[:, None] * Sb[None, :]`'s row factor, leaving a clean
  per-output-channel `Sb` to apply on the vector pipe. This avoids
  needing two extra cube fragments per tile and matches the reference to
  within 5 × 10⁻³ relative error on randomly generated inputs.

- **`hc_split_sinkhorn`** — the GPU implementation runs `pre` (`sigmoid +
  ε`) and `post` (`2 * sigmoid`) heads on one stream and a 20-iteration
  row/col-normalising Sinkhorn on a 4×4 mix matrix on another. The PTO
  port does all three inside one `vector_section`: pre/post are simple
  elementwise blocks; Sinkhorn iterates over `[n, 4, 4]` tiles using
  `row_sum`/`col_sum` + broadcast-divide, with the final row-normalise
  fused into the loop tail. ε is added once after the initial softmax,
  matching the reference order exactly.

- **`sparse_attn`** — pure `vector_section` FlashAttention with online
  streaming softmax. The matmul shapes (`[16, 128] · [128]` per K
  position, K ≤ 128) are too small to amortize cube launch overhead, and
  KV is gathered by arbitrary index so it cannot live in L1 contiguously
  anyway. Per-head softmax stats are stored as full `[H, D]` tiles
  replicated across the D axis to dodge a col-major⇄row-major reshape
  alias that the auto-sync analysis can otherwise miss. KV gather uses
  `pto.load_scalar` of the index → `pto.slice_view` with that dynamic
  row offset → `pto.load` of one `[1, D]` row.
