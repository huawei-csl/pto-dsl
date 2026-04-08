```bash
bash compile.sh           # default matrix size 64 → inverse_lib_64.so
python run_inverse.py     # defaults to --lib-path ./inverse_lib_64.so

bash compile.sh 128       # matrix size 128 → inverse_lib_128.so
python run_inverse.py --matrix-size 128
```

This demo implements one-level 2x2 block inversion for `inv(I + A)` with input shape
`[batch, n, n]`:

- `A` is interpreted as block-lower-triangular:
  `[[A11, 0], [A21, A22]]`, with `A11/A22` size `n/2`.
- `inv(I + A11)` and `inv(I + A22)` are computed by the same fast recurrence used in
  the `basic_dense` / `block_diag` demos.
- `A21` block is recovered by `-inv(I + A22) @ A21 @ inv(I + A11)`.

`run_inverse.py` includes:
- correctness checks on structured random / ill-conditioned generators
- a precision report line in the note style: `c=<n> | error = ...`
