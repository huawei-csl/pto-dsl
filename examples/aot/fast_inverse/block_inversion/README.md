```bash
bash compile.sh           # default matrix size 64
python run_inverse.py

bash compile.sh 128       # another supported matrix size
python run_inverse.py --matrix-size 128 --lib-path ./inverse_lib.so
```

This demo implements one-level 2x2 block inversion for `inv(I + A)` with input shape
`[batch, n, n]`:

- `A` is interpreted as block-lower-triangular:
  `[[A11, 0], [A21, A22]]`, with `A11/A22` size `n/2`.
- `inv(I + A11)` and `inv(I + A22)` are computed by the same fast recurrence used in
  the `basic_dense` / `block_diag` demos.
- `A21` block is recovered by `-inv(I + A22) @ A21 @ inv(I + A11)`.
- launch uses persistent-kernel style with fixed `blockDim=24`; each core loops over
  multiple batch items when `batch > 24`.

`run_inverse.py` includes:
- correctness checks on structured random / ill-conditioned generators
- a precision report line in the note style: `c=<n> | error = ...`
