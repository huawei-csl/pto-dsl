# Educational demo for Sinkhorn-Knopp iteration

```bash
bash ./compile.sh
pytest ./test_sinkhorn.py -v --npu=7
```

`./compile.sh` also builds the hand-written C++ references under `cpp_ref/outputs/kernel_sinkhorn.so` (BATCH=8) and `cpp_ref/outputs/kernel_sinkhorn_v2.so` (interleaved fast path), see `cpp_ref/compile.sh`.

## Bandwidth (large shapes)

Forward pass only; numbers are **median effective GB/s** (decimal 10^9 bytes per second) counting one read plus one write of all 4×4 fp16 matrices—same convention as `pto-kernels-sinkhorn-demo/examples/jit_cpp/fast_hadamard/bench_common.py`.

| Kernel | Source |
|--------|--------|
| **Batched** | `sinkhorn_batch8_builder.py` → `ptoas --enable-insert-sync` (PTODSL) |
| **Naive** | `sinkhorn_k4_builder.py` → `ptoas --enable-insert-sync` (PTODSL) |
| **v2 (PTODSL)** | `sinkhorn_v2_builder.py` → `ptoas --enable-insert-sync` (PTODSL; large-`N` path uses a 32-matrix stack like batched, with `eps` like the demo ref) |
| **C++ ref (v1)** | `cpp_ref/kernel_sinkhorn.cpp` → `cpp_ref/compile.sh` (bisheng, BATCH=8) |
| **C++ ref (v2)** | `cpp_ref/kernel_sinkhorn_v2.cpp` → `cpp_ref/compile.sh` (bisheng, interleaved `TCOLSUM` fast path) |

Run locally (pick a free NPU, e.g. `npu-smi info`). After `./compile.sh`, the C++ `.so` is already in `cpp_ref/outputs/`. Otherwise:

```bash
bash cpp_ref/compile.sh
python3 bench_sinkhorn_bandwidth.py --npu 0
# optional: --shapes 65536,262144 --build-cpp --warmup 8 --iters 24 --no-cpp
```

Sample run on **Ascend 910B2**, `repeat=10`, `warmup=8`, `iters=24`, from `cpp_ref/` (batched / naive / PTODSL v2 use a static unrolled tail, ``repeat=10`` in the builders). Benchmark: `python3 bench_sinkhorn_bandwidth.py --npu …`.

| Shape (matrices) | Batched GB/s | Naive GB/s | v2 Py GB/s | C++ v1 GB/s | C++ v2 GB/s | Batched / naive | v2 Py / C++ v2 |
|------------------|-------------:|-----------:|------------:|------------:|------------:|----------------:|---------------:|
| (1, 65536, 4, 4) — 65536 matrices | 1.515 | 1.162 | 1.712 | 1.430 | 5.934 | 1.30 | 0.29 |
| (1, 262144, 4, 4) — 262144 matrices | 1.596 | 1.183 | 1.794 | 1.502 | 6.997 | 1.35 | 0.26 |

PTODSL **batched** matches the C++ v1 (BATCH=8) algorithm. **v2 (PTODSL)** is a larger stack (32 matrices per chunk) with the same `eps` pattern as batched on the large-`N` path so it tracks `sinkhorn_normalize_ref` in tests; the hand **C++ v2** kernel uses interleaved column reduction and different sync/pipelining, so effective GB/s is not directly comparable column-for-column. Absolute GB/s varies by chip and load.
