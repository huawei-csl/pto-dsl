# fp8_gemm — per-channel fp8 (e4m3) GEMM with fused Sa/Sb scales

PTO DSL port of the deepseek_v4 `fp8_gemm` op. The kernel keeps the
matmul pure (cube fp32 accum → fp16 cast); the per-channel `Sa[m]`
rescale is fused into a host-side pre-scale of `A`, leaving a clean
per-output-channel `Sb` to apply on the vector pipe.

```bash
bash ./compile.sh
python ./run_fp8_gemm.py
```
