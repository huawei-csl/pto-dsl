```bash
bash compile.sh           # default matrix size 64
python run_inverse.py

bash compile.sh 128       # another supported matrix size
python run_inverse.py --matrix-size 128 --lib-path ./inverse_lib.so
```

This dense demo uses input shape `[batch, n, n]` and applies the same fast-inverse recurrence
as the block-diagonal example, with `log2_blocksize = log2(n)` (no extra diagonal block size).
It uses persistent-kernel style launch with fixed `blockDim=24`, and each core loops over
its assigned batch indices at runtime.

For numerical stability in this educational demo, test inputs are generated as:
`M = I + scale * random`, and the kernel computes `inv(M)` via `A = M - I`.
