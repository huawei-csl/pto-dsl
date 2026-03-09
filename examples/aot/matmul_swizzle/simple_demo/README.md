Usage:

```bash
bash ./compile.sh
# Run both variants (default)
python ./run_simple_matmul.py

# Or run a single variant
python ./run_simple_matmul.py --variant auto-sync
python ./run_simple_matmul.py --variant manual-sync

# Benchmark auto-sync vs manual-sync performance.
python ./bench_matmul.py
```
