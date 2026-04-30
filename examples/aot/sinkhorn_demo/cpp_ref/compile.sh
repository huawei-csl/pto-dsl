#!/usr/bin/env bash
# Compile hand-written PTO-ISA Sinkhorn kernel to outputs/kernel_sinkhorn.so
# (same call_sinkhorn ABI as caller_sinkhorn_k4.cpp in the parent demo).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p outputs

PTO_ROOT="${PTO_LIB_PATH:-}"
if [[ -z "${PTO_ROOT}" || ! -f "${PTO_ROOT}/include/pto/pto-inst.hpp" ]]; then
  PTO_ROOT="/workdir/pto-isa-master"
fi
if [[ ! -f "${PTO_ROOT}/include/pto/pto-inst.hpp" ]]; then
  PTO_ROOT="${PTO_LIB_PATH:-/sources/pto-isa}"
fi

bisheng \
  -fPIC -shared -xcce -DMEMORY_BASE \
  -O2 -std=c++17 -Wno-ignored-attributes \
  --cce-aicore-arch=dav-c220-vec \
  "-I${PTO_ROOT}/include" \
  kernel_sinkhorn.cpp \
  -o outputs/kernel_sinkhorn.so

bisheng \
  -fPIC -shared -xcce -DMEMORY_BASE \
  -O2 -std=c++17 -Wno-ignored-attributes \
  --cce-aicore-arch=dav-c220-vec \
  "-I${PTO_ROOT}/include" \
  kernel_sinkhorn_v2.cpp \
  -o outputs/kernel_sinkhorn_v2.so

echo "Built $(pwd)/outputs/kernel_sinkhorn.so (C++ BATCH=8) and $(pwd)/outputs/kernel_sinkhorn_v2.so (C++ v2)"
