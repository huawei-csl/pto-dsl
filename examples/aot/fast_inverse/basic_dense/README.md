```bash
bash compile.sh 64 single   # build single-buffer baseline -> inverse_lib.so
bash compile.sh 64 double   # build double-buffer variant -> inverse_lib_db.so

# Validate correctness for both variants
python run_inverse.py \
  --matrix-size 64 \
  --lib-path ./inverse_lib.so \
  --double-lib-path ./inverse_lib_db.so \
  --variant both

# Single variant only (optional)
python run_inverse.py --matrix-size 128 --variant single --lib-path ./inverse_lib.so
python run_inverse.py --matrix-size 128 --variant double --double-lib-path ./inverse_lib_db.so

# Compare effective bandwidth: single vs double
python bench_inverse.py \
  --matrix-size 64 \
  --lib-path ./inverse_lib.so \
  --double-lib-path ./inverse_lib_db.so \
  --out-png bench_inverse_bandwidth.png
```

`bench_inverse.py` reports and plots bandwidth for both variants using only:
- read of `in_delta` (`torch_to_ctypes(in_delta)`)
- write of `out` (`torch_to_ctypes(out)`)

Timing measures only the kernel launch (`lib.call_kernel(...)`) and excludes tensor
preparation (`identity`, `in_delta`, `identity_neg`, `out` creation).

This dense demo uses input shape `[batch, n, n]` and applies the same fast-inverse recurrence
as the block-diagonal example, with `log2_blocksize = log2(n)` (no extra diagonal block size).
It uses persistent-kernel style launch with fixed `blockDim=24`, and each core loops over
its assigned batch indices at runtime.

Double-buffering follows the same ping-pong idea used in the matmul optimization guide:
- keep two `L1` buffers for recurrence states (`X`, `Y`)
- keep two `L0` operand buffers (`A`, `B`)
- alternate odd/even buffers each iteration

For `n=128`, this stays within the documented SRAM constraints:
- `L1`: `2*X + 2*Y + I` = `5 * (128*128*2 B)` = `160 KiB` < `512 KiB`
- `L0A`: `2 * (128*128*2 B)` = `64 KiB` (at limit)
- `L0B`: `2 * (128*128*2 B)` = `64 KiB` (at limit)
- `L0C`: `1 * (128*128*4 B)` = `64 KiB` < `128 KiB`

For numerical stability in this educational demo, test inputs are generated as:
`M = I + scale * random`, and the kernel computes `inv(M)` via `A = M - I`.
