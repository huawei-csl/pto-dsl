```bash
bash ./compile.sh
python ./run_matmul.py
```

`run_matmul.py` runs the broader reference-style shape sweep:
- `block_dim`: `1, 20, 24`
- `M`: `128, 640, ..., 4224`
- `(N, K)`: `(4096, 4096)`, `(8192, 8192)`, `(16384, 16384)`

It prints the worst error per shape and fails if the global worst case exceeds the configured thresholds.
