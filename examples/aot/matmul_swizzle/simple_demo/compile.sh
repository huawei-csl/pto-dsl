#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="./build_artifacts"
mkdir -p "${ARTIFACT_DIR}"

rm -f "${ARTIFACT_DIR}"/*.pto "${ARTIFACT_DIR}"/*.cpp "${ARTIFACT_DIR}"/*.so

# Manual-sync kernel variant: explicit record/wait events in PTO.
python ./simple_matmul_builder.py --manual-sync > "${ARTIFACT_DIR}/simple_matmul_manual_sync.pto"
ptoas "${ARTIFACT_DIR}/simple_matmul_manual_sync.pto" -o "${ARTIFACT_DIR}/simple_matmul_manual_sync.cpp"

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"${ARTIFACT_DIR}/simple_matmul_manual_sync.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt \
    ./caller.cpp \
    -o "${ARTIFACT_DIR}/simple_matmul_manual_sync_kernel.so"

# Auto-sync kernel variant: no explicit record/wait events in PTO.
python ./simple_matmul_builder.py > "${ARTIFACT_DIR}/simple_matmul_auto_sync.pto"
ptoas --enable-insert-sync "${ARTIFACT_DIR}/simple_matmul_auto_sync.pto" -o "${ARTIFACT_DIR}/simple_matmul_auto_sync.cpp"

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"${ARTIFACT_DIR}/simple_matmul_auto_sync.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt_autosync \
    ./caller.cpp \
    -o "${ARTIFACT_DIR}/simple_matmul_auto_sync_kernel.so"

# Auto-sync kernel variant without swizzle.
python ./simple_matmul_builder.py --disable-swizzle > "${ARTIFACT_DIR}/simple_matmul_auto_sync_noswizzle.pto"
ptoas --enable-insert-sync "${ARTIFACT_DIR}/simple_matmul_auto_sync_noswizzle.pto" -o "${ARTIFACT_DIR}/simple_matmul_auto_sync_noswizzle.cpp"

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"${ARTIFACT_DIR}/simple_matmul_auto_sync_noswizzle.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt_autosync \
    ./caller.cpp \
    -o "${ARTIFACT_DIR}/simple_matmul_auto_sync_noswizzle_kernel.so"

# Single-buffer auto-sync variant: simplified no ping-pong buffers, baseline non-swizzle.
python ./single_buffer_matmul.py > "${ARTIFACT_DIR}/single_buffer_matmul_auto_sync.pto"
ptoas --enable-insert-sync "${ARTIFACT_DIR}/single_buffer_matmul_auto_sync.pto" -o "${ARTIFACT_DIR}/single_buffer_matmul_auto_sync.cpp"

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"${ARTIFACT_DIR}/single_buffer_matmul_auto_sync.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt_single_buffer_autosync \
    ./caller.cpp \
    -o "${ARTIFACT_DIR}/single_buffer_matmul_auto_sync_kernel.so"
