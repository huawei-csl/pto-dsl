# Educational demo for Sinkhorn-Knopp iteration

```bash
bash ./compile.sh
pytest ./test_sinkhorn.py -v --npu=7
```

`./compile.sh` also builds the hand-written C++ reference under `cpp_ref/outputs/kernel_sinkhorn.so` (see `cpp_ref/compile.sh`).

## Bandwidth (large shapes)

Forward pass only; numbers are **median effective GB/s** (decimal 10^9 bytes per second) counting one read plus one write of all 4×4 fp16 matrices—same convention as `pto-kernels-sinkhorn-demo/examples/jit_cpp/fast_hadamard/bench_common.py`.

| Kernel | Source |
|--------|--------|
| **Batched** | `sinkhorn_batch8_builder.py` → `ptoas --enable-insert-sync` (PTODSL) |
| **Naive** | `sinkhorn_k4_builder.py` → `ptoas --enable-insert-sync` (PTODSL) |
| **C++ ref** | `cpp_ref/kernel_sinkhorn.cpp` → `cpp_ref/compile.sh` (bisheng, manual sync) |

Run locally (pick a free NPU, e.g. `npu-smi info`). After `./compile.sh`, the C++ `.so` is already in `cpp_ref/outputs/`. Otherwise:

```bash
bash cpp_ref/compile.sh
python3 bench_sinkhorn_bandwidth.py --npu 0
# optional: --shapes 65536,262144 --build-cpp --warmup 8 --iters 24 --no-cpp
```

Sample run on **Ascend 910B2**, `repeat=10`, `warmup=8`, `iters=24`, C++ ref from `cpp_ref/`:

| Shape (matrices) | Batched GB/s | Naive GB/s | C++ ref GB/s | Batched / naive | Batched / C++ |
|------------------|-------------:|-----------:|-------------:|----------------:|--------------:|
| (1, 65536, 4, 4) — 65536 matrices | 1.508 | 1.162 | 1.434 | 1.30 | 1.05 |
| (1, 262144, 4, 4) — 262144 matrices | 1.593 | 1.185 | 1.495 | 1.34 | 1.07 |

PTODSL batched matches the C++ reference algorithm (BATCH=8 stack); effective GB/s can differ slightly from the hand kernel (auto-inserted sync, codegen). Absolute GB/s varies by chip and load.
