# TileLang Sinkhorn baseline

Self-contained TileLang (Ascend NPU) Sinkhorn normalization: stable softmax on the last axis, add `eps`, then alternating row/column scaling for `repeat` steps (reference helper `sinkhorn_normalize_ref` in `sinkhorn_kernel.py`).

## Requirements

- Python 3 with `torch` and `torch_npu`
- `tilelang` built with Ascend NPU codegen enabled

## Commands

```bash
cd tilelang_baseline
python3 -m pytest test_sinkhorn.py -q
python3 bench_sinkhorn.py --npu 0 --warmup 8 --iters 24 --repeat 10 --hc 4 --shapes 65536,262144
```

## Implementation notes

- Inner TileLang steps follow the usual Sinkhorn pattern: numerically stable row-wise softmax, one column normalization, then repeated row/column normalizations (implemented with `T.reduce_*` and `T.tile.*` ops on a small `(hc, hc)` tile).
- Each kernel launch handles at most **131070** matrices (`ceil(n/2)` grid slots). Larger batch counts are split across launches (`sinkhorn_normalize_tilelang`).

## Sample bandwidth (this environment)

Measured with `bench_sinkhorn.py` defaults (`repeat=10`, float32, effective GB/s = read+write tensor bytes / median latency):

| Shape `(1, n1, 4, 4)` | Matrices | Median GB/s |
|----------------------|----------|------------:|
| n1 = 65536           | 65536    | 1.009       |
| n1 = 262144          | 262144   | 1.067       |

Hardware and driver versions differ; treat these as indicative only.
