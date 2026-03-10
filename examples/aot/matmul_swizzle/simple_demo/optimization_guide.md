# Matmul Optimization Guide (4 Steps)

This folder is organized as a tutorial-style optimization path for dynamic-shape matmul.

## Step Overview

1. `step1_baseline.py`
   - Functionally correct dynamic-shape matmul.
   - No double-buffer, no swizzle, no manual software pipelining.
2. `step2_doublebuffer.py`
   - Adds double-buffer only.
   - Still no swizzle, uses auto-sync insertion.
3. `step3_swizzle.py`
   - Adds swizzle on top of step2.
   - Uses double-buffer + auto-sync.
4. `step4_manual_pipelining.py`
   - Adds manual software pipelining on top of step3.
   - Uses explicit event record/wait.

Shared pieces are extracted into `common_utils.py`:
- metadata/type/buffer definitions
- `swizzle_nz(...)`

## Build All Steps

```bash
bash ./compile.sh
```

Artifacts are emitted into `build_artifacts/`:
- `step1_baseline_kernel.so`
- `step2_doublebuffer_kernel.so`
- `step3_swizzle_kernel.so`
- `step4_manual_pipelining_kernel.so`

## Reproducible Per-Step Build Commands

```bash
ARTIFACT_DIR=./build_artifacts
mkdir -p "${ARTIFACT_DIR}"

# Step1 baseline
python ./step1_baseline.py > "${ARTIFACT_DIR}/step1_baseline.pto"
ptoas --enable-insert-sync "${ARTIFACT_DIR}/step1_baseline.pto" -o "${ARTIFACT_DIR}/step1_baseline.cpp"
bisheng -fPIC -shared -xcce -O2 -std=c++17 --npu-arch=dav-2201 -DMEMORY_BASE \
  -I"${ASCEND_TOOLKIT_HOME}/include" \
  -DKERNEL_CPP="\"${ARTIFACT_DIR}/step1_baseline.cpp\"" \
  -DKERNEL_FN=matmul_kernel_step1_baseline \
  ./caller.cpp -o "${ARTIFACT_DIR}/step1_baseline_kernel.so"

# Step2 double-buffer only
python ./step2_doublebuffer.py --disable-swizzle > "${ARTIFACT_DIR}/step2_doublebuffer.pto"
ptoas --enable-insert-sync "${ARTIFACT_DIR}/step2_doublebuffer.pto" -o "${ARTIFACT_DIR}/step2_doublebuffer.cpp"
bisheng -fPIC -shared -xcce -O2 -std=c++17 --npu-arch=dav-2201 -DMEMORY_BASE \
  -I"${ASCEND_TOOLKIT_HOME}/include" \
  -DKERNEL_CPP="\"${ARTIFACT_DIR}/step2_doublebuffer.cpp\"" \
  -DKERNEL_FN=matmul_kernel_ABt_autosync \
  ./caller.cpp -o "${ARTIFACT_DIR}/step2_doublebuffer_kernel.so"

# Step3 swizzle + double-buffer
python ./step3_swizzle.py > "${ARTIFACT_DIR}/step3_swizzle.pto"
ptoas --enable-insert-sync "${ARTIFACT_DIR}/step3_swizzle.pto" -o "${ARTIFACT_DIR}/step3_swizzle.cpp"
bisheng -fPIC -shared -xcce -O2 -std=c++17 --npu-arch=dav-2201 -DMEMORY_BASE \
  -I"${ASCEND_TOOLKIT_HOME}/include" \
  -DKERNEL_CPP="\"${ARTIFACT_DIR}/step3_swizzle.cpp\"" \
  -DKERNEL_FN=matmul_kernel_ABt_autosync \
  ./caller.cpp -o "${ARTIFACT_DIR}/step3_swizzle_kernel.so"

# Step4 manual software pipelining
python ./step4_manual_pipelining.py > "${ARTIFACT_DIR}/step4_manual_pipelining.pto"
ptoas "${ARTIFACT_DIR}/step4_manual_pipelining.pto" -o "${ARTIFACT_DIR}/step4_manual_pipelining.cpp"
bisheng -fPIC -shared -xcce -O2 -std=c++17 --npu-arch=dav-2201 -DMEMORY_BASE \
  -I"${ASCEND_TOOLKIT_HOME}/include" \
  -DKERNEL_CPP="\"${ARTIFACT_DIR}/step4_manual_pipelining.cpp\"" \
  -DKERNEL_FN=matmul_kernel_ABt \
  ./caller.cpp -o "${ARTIFACT_DIR}/step4_manual_pipelining_kernel.so"
```

## Correctness and Benchmark

```bash
# Validate numerical correctness for all 4 steps
python ./run_simple_matmul.py

# Optional: run one step only
python ./run_simple_matmul.py --variant step1-baseline
python ./run_simple_matmul.py --variant step2-doublebuffer
python ./run_simple_matmul.py --variant step3-swizzle
python ./run_simple_matmul.py --variant step4-manual-pipelining

# Run stepwise performance comparison
python ./bench_matmul.py
```

## Reference Performance (Example)

```text
=== Summary ===
Step1 (double-buffer speedup, both non-swizzle auto-sync):
avg FLOP ratio(double_noswizzle_auto/single_noswizzle): 1.607x
min FLOP ratio(double_noswizzle_auto/single_noswizzle): 0.943x
max FLOP ratio(double_noswizzle_auto/single_noswizzle): 1.826x
Step2 (swizzle speedup, both double-buffer auto-sync):
avg FLOP ratio(double_swizzle_auto/double_noswizzle_auto): 1.227x
min FLOP ratio(double_swizzle_auto/double_noswizzle_auto): 0.863x
max FLOP ratio(double_swizzle_auto/double_noswizzle_auto): 1.871x
Step3 (manual-sync speedup, both double-buffer swizzle):
avg FLOP ratio(double_swizzle_manual/double_swizzle_auto): 1.100x
min FLOP ratio(double_swizzle_manual/double_swizzle_auto): 1.001x
max FLOP ratio(double_swizzle_manual/double_swizzle_auto): 1.173x
```
