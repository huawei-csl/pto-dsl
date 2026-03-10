#!/usr/bin/env bash
set -euo pipefail

rm -f \
    simple_matmul_auto_sync.pto simple_matmul_manual_sync.pto \
    simple_matmul_auto_sync.cpp simple_matmul_manual_sync.cpp \
    simple_matmul_auto_sync_kernel.so simple_matmul_manual_sync_kernel.so \
    single_buffer_matmul_auto_sync.pto single_buffer_matmul_auto_sync.cpp \
    single_buffer_matmul_auto_sync_kernel.so

# Manual-sync kernel variant: explicit record/wait events in PTO.
python ./simple_matmul_builder.py --manual-sync > simple_matmul_manual_sync.pto
ptoas simple_matmul_manual_sync.pto -o simple_matmul_manual_sync.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"simple_matmul_manual_sync.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt \
    ./caller.cpp \
    -o ./simple_matmul_manual_sync_kernel.so

# Auto-sync kernel variant: no explicit record/wait events in PTO.
python ./simple_matmul_builder.py > simple_matmul_auto_sync.pto
ptoas --enable-insert-sync simple_matmul_auto_sync.pto -o simple_matmul_auto_sync.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"simple_matmul_auto_sync.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt_autosync \
    ./caller.cpp \
    -o ./simple_matmul_auto_sync_kernel.so

# Single-buffer auto-sync variant: simplified no ping-pong buffers.
python ./single_buffer_matmul.py > single_buffer_matmul_auto_sync.pto
ptoas --enable-insert-sync single_buffer_matmul_auto_sync.pto -o single_buffer_matmul_auto_sync.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"single_buffer_matmul_auto_sync.cpp\"" \
    -DKERNEL_FN=matmul_kernel_ABt_single_buffer_autosync \
    ./caller.cpp \
    -o ./single_buffer_matmul_auto_sync_kernel.so
