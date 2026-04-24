# Educational demo for Sinkhorn-Knopp iteration

```bash
bash ./compile.sh
pytest ./test_sinkhorn.py -v --npu=7
```

## Bandwidth (large shapes)

Forward pass only; numbers are **median effective GB/s** (decimal 10^9 bytes per second) counting one read plus one write of all 4×4 fp16 matrices—same convention as `pto-kernels-sinkhorn-demo/examples/jit_cpp/fast_hadamard/bench_common.py`. **Batched** is `sinkhorn_batch8_builder.py` (stack load); **naive** is `sinkhorn_k4_builder.py`.

Run locally (pick a free NPU, e.g. `npu-smi info`):

```bash
python3 bench_sinkhorn_bandwidth.py --npu 0
# optional: --shapes 65536,262144,1048576 --warmup 8 --iters 24
```

Sample run on **Ascend 910B2**, `repeat=10`, `warmup=8`, `iters=24`:

| Shape (matrices) | Batched GB/s | Naive GB/s | Batched / naive |
|------------------|-------------:|-----------:|----------------:|
| (1, 65536, 4, 4) — 65536 matrices | 1.511 | 1.162 | 1.30 |
| (1, 262144, 4, 4) — 262144 matrices | 1.597 | 1.185 | 1.35 |

Absolute GB/s depends on chip, driver, and load; the ratio is the usual takeaway for this demo.
