# fp8_gemm — PTO DSL port

Original (TileLang/GPU): block-scaled FP8 GEMM,
`C[m,n] = sum_k A_fp8[m,k] * B_fp8[k,n] * Sa[m, k_g] * Sb[k_g, n_b]`,
where the per-128-block scales `Sa` and `Sb` are required to recover the
mantissa precision lost to FP8's tiny dynamic range.

## NPU port: host-side pre-scale

PTO DSL on Ascend exposes FP16 / FP32 (no FP8). FP16's ±65504 dynamic
range comfortably accommodates the pre-scaled values for any realistic
`Sa, Sb`, so we absorb both scales into `A` and `B` host-side before
launching the kernel:

```
A_scaled[m, k] = A[m, k] * Sa[m, k_g]                  (row-broadcast)
B_scaled[k, n] = B[k, n] * Sb[k_g, n_b]                (col-broadcast)
C[m, n]        = sum_k A_scaled[m, k] * B_scaled[k, n]
```

This is mathematically identical to the GPU per-K-group fusion. The
kernel itself is a clean FP16 GEMM with FP32 accumulator; `Sa, Sb` stay
in the kernel signature for API parity but are not read on-device.

## Status

Tests use non-trivial random `Sa, Sb` (lognormal, ~exp(N(0,1))) to
exercise the full scale-fusion path end-to-end, plus a unit-scale
sanity test. 5/5 pass.

## Why not on-device fusion?

Fusing scales on-device requires a vector pipe pass over the FP32 cube
accumulator per K-group, which means a multifunc cube_kernel ↔
vector_kernel design with bidirectional FIFO sync (see
[examples/aot/flash_attention_manual_sync](../../flash_attention_manual_sync/)).
That's the right pattern when the scales themselves are computed on-chip
(e.g. fused per-tile dynamic quant), but for static scales it adds ~250
LoC of sync plumbing for zero numerical benefit on FP16 inputs.
