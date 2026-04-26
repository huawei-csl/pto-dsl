# DeepSeek-V4 PTO ports — overview

> This file is intentionally **not named `README.md`** so that
> [`validate_all_examples.py`](../../validate_all_examples.py) walks
> into each kernel sub-directory directly instead of trying to run a
> repo-level recipe from here.

PTO DSL ports of the six custom kernels used by the DeepSeek-V4
reference implementation. Every kernel is self-contained in its own
folder and follows the standard examples-tree workflow:

1. `bash ./compile.sh` — emits `.pto` → `.cpp` → `*_lib.so`.
2. `python ./run_*.py` — runs the kernel on NPU and asserts numerical
   equivalence with a PyTorch reference (exits non-zero on mismatch).
3. (optional) `python ./bench_*.py` — microbenchmarks vs PyTorch
   baselines (only `sparse_attn/` and `hc_split_sinkhorn/`).

## Kernels

| Folder | What it does | Pipe(s) |
|---|---|---|
| [act_quant/](act_quant/) | Per-row absmax fp16 → int8 quant (`max(|x|)/127`, `round(x/scale)`) | vector |
| [fp4_act_quant/](fp4_act_quant/) | Per-row fp16 → mxfp4 (e2m1) quant with shared exponent + lookup-table cast | vector |
| [fp8_gemm/](fp8_gemm/) | Per-channel fp8 (e4m3) GEMM with host-side fused `Sa`/`Sb` pre-scale | cube + vector |
| [fp4_gemm/](fp4_gemm/) | Per-channel fp4 (e2m1) GEMM with host-side fused `Sa`/`Sb` pre-scale | cube + vector |
| [hc_split_sinkhorn/](hc_split_sinkhorn/) | Fused MoE-router head: pre/post sigmoid + 20-iter Sinkhorn, all on-device | vector |
| [sparse_attn/](sparse_attn/) | FlashAttention with indexed top-k KV gather + per-head sink logit | vector |

## Run a single kernel

```bash
cd examples/aot/deepseek_v4/sparse_attn
bash ./compile.sh
python ./run_sparse_attn.py
```

The generated `.pto`, `.cpp`, `.so` files are gitignored.

## Run all of them

From the repo root:

```bash
python examples/validate_all_examples.py
```

This walks every `README.md` under `examples/`, runs the bash block in
each, and reports pass/fail. The deepseek_v4 kernels appear in the
listing as e.g. `aot/deepseek_v4/sparse_attn`.

## Sample bench output

`sparse_attn/`, vs `torch.gather` + `npu_fused_infer_attention_score`
(MQA mode, sink logit dropped — speed baseline only):

```
  B   M     N    K     pto us     ref us   fused us   pto/ref  pto/fused
------------------------------------------------------------------------
  1   1   128   64     161.15     533.05     265.03     3.31x      1.64x
  1   4   256  128     209.56    1692.93     252.36     8.08x      1.20x
  4   4  1024  128     207.77    6071.60     246.57    29.22x      1.19x
  8   8  2048  128     304.49   24658.49     244.67    80.98x      0.80x
```

`hc_split_sinkhorn/`, vs eager PyTorch reference:

```
      n     pto us     ref us  speedup
----------------------------------------
     64     173.27    2803.42   16.18x
   1024     218.70    2761.33   12.63x
  16384    1786.32    2741.09    1.53x
```

## Implementation notes

- **`fp8_gemm` / `fp4_gemm`** — the GPU op fuses an outer `Sa[m] * Sb[n]`
  per-channel rescale into the GEMM. The PTO kernels keep the matmul
  pure (cube fp32 accum → fp16 cast) and instead **pre-scale `A` on the
  host** by the per-row factor, leaving a clean per-output-channel `Sb`
  to apply on the vector pipe. Avoids two extra cube fragments per tile
  and matches reference within 5 × 10⁻³ relative error.
- **`hc_split_sinkhorn`** — all three router heads (pre / post / 20-iter
  Sinkhorn over `[n, 4, 4]`) run inside one `vector_section`. ε is added
  once after the initial softmax to match the reference order exactly.
- **`sparse_attn`** — pure `vector_section` FlashAttention with online
  streaming softmax. The matmul shapes (`[16, 128] · [128]` per K
  position, K ≤ 128) are too small to amortize cube launch overhead, and
  KV is gathered by arbitrary index so it cannot live in L1 contiguously
  anyway. Per-head softmax stats are stored as full `[H, D]` tiles
  replicated across the D axis to dodge a col-major⇄row-major reshape
  alias that auto-sync analysis can otherwise miss. KV gather uses
  `pto.load_scalar` of the index → `pto.slice_view` with that dynamic
  row offset → `pto.load` of one `[1, D]` row.
