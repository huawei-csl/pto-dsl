# fp4_gemm — per-channel fp4 (e2m1) GEMM with fused Sa/Sb scales

PTO DSL port of the deepseek_v4 `fp4_gemm` op. Same scale-fusion design
as `fp8_gemm` (host-side pre-scale of `A`, vector-pipe `Sb`), but with
fp4 (e2m1) weights using a per-block-K group of 32 elements.

```bash
bash ./compile.sh
python ./run_fp4_gemm.py
```
