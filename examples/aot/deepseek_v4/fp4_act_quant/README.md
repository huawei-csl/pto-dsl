# fp4_act_quant — fp16 → mxfp4 (e2m1) per-block quantization

PTO DSL port of the deepseek_v4 `fp4_act_quant` op. Per `BLOCK_SIZE`
group of 32 elements computes a shared exponent scale and casts each
value through a fp4 (e2m1) lookup table.

```bash
bash ./compile.sh
python ./run_fp4_act_quant.py
```
