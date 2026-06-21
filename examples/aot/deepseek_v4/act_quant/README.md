# act_quant — fp16 → int8 per-row absmax quantization

PTO DSL port of the deepseek_v4 `act_quant` op. Per row computes
`scale = max(|x|) / 127`, then `y = round(x / scale)`.

```bash
bash ./compile.sh
python ./run_act_quant.py
```
