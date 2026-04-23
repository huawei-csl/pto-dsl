#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p outputs
python3 sinkhorn_k4_builder.py > outputs/sinkhorn_k4.pto
ptoas --enable-insert-sync outputs/sinkhorn_k4.pto -o outputs/sinkhorn_k4_generated.cpp

PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
bisheng \
  -fPIC -shared -xcce -DMEMORY_BASE \
  -O2 -std=c++17 -Wno-ignored-attributes \
  --cce-aicore-arch=dav-c220-vec \
  -isystem "${PTO_LIB_PATH}/include" \
  caller_sinkhorn_k4.cpp \
  -o outputs/kernel_sinkhorn.so

echo "Built outputs/kernel_sinkhorn.so"
