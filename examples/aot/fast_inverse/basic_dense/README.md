```bash
bash compile.sh           # default matrix size 64
python run_inverse.py

bash compile.sh 128       # another supported matrix size
python run_inverse.py --matrix-size 128 --lib-path ./inverse_lib.so

# Plot: batch size vs bandwidth (GiB/s)
python bench_inverse.py \
  --matrix-size 64 \
  --lib-path ./inverse_lib.so \
  --out-png bench_inverse_bandwidth.png
```

`bench_inverse.py` reports and plots bandwidth using only:
- read of `in_delta` (`torch_to_ctypes(in_delta)`)
- write of `out` (`torch_to_ctypes(out)`)

Timing measures only the kernel launch (`lib.call_kernel(...)`) and excludes tensor
preparation (`identity`, `in_delta`, `identity_neg`, `out` creation).

This dense demo uses input shape `[batch, n, n]` and applies the same fast-inverse recurrence
as the block-diagonal example, with `log2_blocksize = log2(n)` (no extra diagonal block size).
It uses persistent-kernel style launch with fixed `blockDim=24`, and each core loops over
its assigned batch indices at runtime.

For numerical stability in this educational demo, test inputs are generated as:
`M = I + scale * random`, and the kernel computes `inv(M)` via `A = M - I`.
