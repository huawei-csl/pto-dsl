#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="./build_artifacts"
mkdir -p "${ARTIFACT_DIR}"

rm -f "${ARTIFACT_DIR}"/*.pto "${ARTIFACT_DIR}"/*.cpp "${ARTIFACT_DIR}"/*.so

# CANN 8.5 headers don't have CompactMode, need latest pto-isa source
PTO_LIB_PATH=/sources/pto-isa

# Step1 baseline: functionally correct dynamic-shape matmul without optimizations.
python ./step1_baseline.py > "${ARTIFACT_DIR}/step1_baseline.pto"
ptoas --enable-insert-sync "${ARTIFACT_DIR}/step1_baseline.pto" -o "${ARTIFACT_DIR}/step1_baseline.cpp"

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${PTO_LIB_PATH}/include" \
    -DKERNEL_CPP="\"${ARTIFACT_DIR}/step1_baseline.cpp\"" \
    -DKERNEL_FN=matmul_kernel_step1_baseline \
    ./caller.cpp \
    -o "${ARTIFACT_DIR}/step1_baseline_kernel.so"

# Step2: double-buffer only (no swizzle, auto-sync).
python ./step2_doublebuffer.py > "${ARTIFACT_DIR}/step2_doublebuffer.pto"
ptoas --enable-insert-sync "${ARTIFACT_DIR}/step2_doublebuffer.pto" -o "${ARTIFACT_DIR}/step2_doublebuffer.cpp"

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${PTO_LIB_PATH}/include" \
    -DKERNEL_CPP="\"${ARTIFACT_DIR}/step2_doublebuffer.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt_autosync \
    ./caller.cpp \
    -o "${ARTIFACT_DIR}/step2_doublebuffer_kernel.so"

# Step3: swizzle + double-buffer (auto-sync).
python ./step3_swizzle.py > "${ARTIFACT_DIR}/step3_swizzle.pto"
ptoas --enable-insert-sync "${ARTIFACT_DIR}/step3_swizzle.pto" -o "${ARTIFACT_DIR}/step3_swizzle.cpp"

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${PTO_LIB_PATH}/include" \
    -DKERNEL_CPP="\"${ARTIFACT_DIR}/step3_swizzle.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt_autosync \
    ./caller.cpp \
    -o "${ARTIFACT_DIR}/step3_swizzle_kernel.so"

# Step4: swizzle + double-buffer + manual software pipelining.
python ./step4_manual_pipelining.py > "${ARTIFACT_DIR}/step4_manual_pipelining.pto"
ptoas "${ARTIFACT_DIR}/step4_manual_pipelining.pto" -o "${ARTIFACT_DIR}/step4_manual_pipelining.cpp"

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${PTO_LIB_PATH}/include" \
    -DKERNEL_CPP="\"${ARTIFACT_DIR}/step4_manual_pipelining.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt \
    ./caller.cpp \
    -o "${ARTIFACT_DIR}/step4_manual_pipelining_kernel.so"
