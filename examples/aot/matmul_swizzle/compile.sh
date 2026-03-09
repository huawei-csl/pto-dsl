#!/usr/bin/env bash
set -euo pipefail

rm -f \
    matmul_auto_sync.pto matmul_manual_sync.pto \
    matmul_auto_sync.cpp matmul_manual_sync.cpp \
    matmul_auto_sync_lib.so matmul_manual_sync_lib.so

# Auto-sync path: rely on ptoas synchronization insertion.
python ./matmul_builder.py > ./matmul_auto_sync.pto
ptoas --enable-insert-sync ./matmul_auto_sync.pto -o ./matmul_auto_sync.cpp

# Manual-sync path: explicit record/wait events from builder.
python ./matmul_builder.py --manual-sync > ./matmul_manual_sync.pto
ptoas ./matmul_manual_sync.pto -o ./matmul_manual_sync.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"matmul_auto_sync.cpp\"" \
    ./caller.cpp \
    -o ./matmul_auto_sync_lib.so

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"matmul_manual_sync.cpp\"" \
    ./caller.cpp \
    -o ./matmul_manual_sync_lib.so
