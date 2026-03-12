Usage:

```bash
# Build all tutorial steps
bash ./compile.sh

# Run correctness on all steps (default)
python ./run_matmul.py

# Or run one specific tutorial step
python ./run_matmul.py --variant step1-baseline
python ./run_matmul.py --variant step2-doublebuffer
python ./run_matmul.py --variant step3-swizzle
python ./run_matmul.py --variant step4-manual-pipelining

# Stepwise benchmark comparisons:
# Step1: double-buffer vs single-buffer (both non-swizzle, auto-sync)
# Step2: swizzle vs non-swizzle (both double-buffer, auto-sync)
# Step3: manual-sync vs auto-sync (both double-buffer, swizzle)
python ./bench_matmul.py
```

See `optimization_guide.md` for full step-by-step build and benchmark commands.
