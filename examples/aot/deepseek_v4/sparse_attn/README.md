# sparse_attn — FlashAttention with indexed top-k KV gather

PTO DSL port of the deepseek_v4 `sparse_attn` op. Pure
`vector_section` FlashAttention: per (batch, query) the kernel gathers
K KV rows by index and runs an online streaming softmax with a
per-head additive sink logit folded into the denominator (and dropped
from the V mix, matching the reference).

```bash
bash ./compile.sh
python ./run_sparse_attn.py
python ./bench_sparse_attn.py
```
