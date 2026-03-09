#!/usr/bin/env bash
set -euo pipefail

rm -f simple_matmul.pto simple_matmul.cpp simple_matmul_kernel.so

python ./simple_matmul_builder.py > simple_matmul.pto
ptoas simple_matmul.pto -o simple_matmul.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    -DKERNEL_CPP="\"simple_matmul.cpp\"" \
    ./caller.cpp \
    -o ./simple_matmul_kernel.so
