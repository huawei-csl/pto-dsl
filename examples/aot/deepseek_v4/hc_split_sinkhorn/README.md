# hc_split_sinkhorn — fused MoE-router head

PTO DSL port of the deepseek_v4 `hc_split_sinkhorn` op. One
`vector_section` runs three heads in fusion: `pre = sigmoid(...) + ε`,
`post = 2 * sigmoid(...)`, and a 20-iter row/col-normalising Sinkhorn
over a `[n, 4, 4]` mix tensor.

```bash
bash ./compile.sh
python ./run_hc_split_sinkhorn.py
python ./bench_hc_split_sinkhorn.py
```
