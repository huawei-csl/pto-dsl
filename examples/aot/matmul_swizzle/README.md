Usage:

```bash
bash ./compile.sh

# Auto-sync (default)
python ./run_matmul.py
python ./bench_matmul.py

# Manual-sync
python ./run_matmul.py --manual-sync
python ./bench_matmul.py --manual-sync
```

Benchmark outputs:
- CSV: `outputs/csv/bench_matmul.csv`
- Optional plots (if `matplotlib` is installed): `outputs/plots/flops_n{N}_k{K}.png`

Useful benchmark options:

```
python ./bench_matmul.py --csv outputs/csv/my_bench.csv --plot-dir outputs/plots
python ./bench_matmul.py --m-list 512,1024,2048,4096
python ./bench_matmul.py --warmup 10 --repeat 50
python ./bench_matmul.py --lib ./matmul_auto_sync_lib.so
python ./bench_matmul.py --lib ./matmul_manual_sync_lib.so
```
