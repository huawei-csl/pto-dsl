#!/usr/bin/env bash
set -euo pipefail

rm -f matmul_autoaddr.pto matmul_autoaddr.cpp matmul_kernel.so

python ./matmul_builder_autoaddr.py > matmul_autoaddr.pto
ptoas --pto-level=level3 matmul_autoaddr.pto -o matmul_autoaddr.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"matmul_autoaddr.cpp\"" \
    ./caller.cpp \
    -o ./matmul_kernel.so
