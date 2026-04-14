Usage:

```bash
bash ./compile.sh
python ./run_hadamard.py
python ./run_hadamard.py --manual-sync
python ./run_hadamard_quant.py
python ./run_hadamard_quant.py --manual-sync
python ./plot_perf.py
```

`run_hadamard.py` tests and benchmarks the plain fast-Hadamard transform (fp16, in-place).

`run_hadamard_quant.py` tests and benchmarks the fused Hadamard+quantize kernel (fp16 → int8),
including uniform quantization and per-group scales/offsets.

Options:

- `--manual-sync`: select the manually-synchronized library variant
- `--block-dim N`: override kernel blockDim (default: 24)
- `--group-size N` (quant only): group size the per-group library was compiled with (default: 128)
