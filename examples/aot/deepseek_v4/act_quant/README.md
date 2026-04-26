# act_quant — PTO DSL port

Original (TileLang/GPU): block-wise BF16 → FP8(e4m3) per-block scaled
quantization, with optional FP32 / E8M0 scale and `inplace` fused dequant.

PTO DSL is an Ascend-NPU DSL with no native FP8 / BF16 / E8M0 support, so
this port substitutes:

| GPU dtype       | NPU port      |
|-----------------|---------------|
| BF16 input      | FP16 input    |
| FP8 e4m3 output | int8 output   |
| FP32 / E8M0 s   | FP32 scale    |

Layout matches the original kernel (per-row absmax over groups of
`block_size` along the last axis), so the algorithm is preserved.
Round-to-power-of-2 (`scale_fmt="ue8m0"`) and `inplace` modes are not
implemented (would require fp8/bf16 dtype support).

Build: `bash compile.sh`
