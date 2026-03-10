Usage:

```bash
bash ./compile.sh
# Run all 4 correctness cases (default)
python ./run_simple_matmul.py

# Or run one specific case
python ./run_simple_matmul.py --variant single-auto-noswizzle
python ./run_simple_matmul.py --variant double-auto-noswizzle
python ./run_simple_matmul.py --variant double-auto-swizzle
python ./run_simple_matmul.py --variant double-manual-swizzle

# Stepwise benchmark comparisons:
# Step1: double-buffer vs single-buffer (both non-swizzle, auto-sync)
# Step2: swizzle vs non-swizzle (both double-buffer, auto-sync)
# Step3: manual-sync vs auto-sync (both double-buffer, swizzle)
python ./bench_matmul.py
```
