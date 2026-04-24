#!/usr/bin/env bash
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

python3 sinkhorn_batch8_builder.py > outputs/sinkhorn_batch8.pto
ptoas --enable-insert-sync outputs/sinkhorn_batch8.pto -o outputs/sinkhorn_batch8_generated.cpp

python3 sinkhorn_k4_builder.py > outputs/sinkhorn_k4.pto
ptoas --enable-insert-sync outputs/sinkhorn_k4.pto -o outputs/sinkhorn_k4_generated.cpp

BFLAGS=(
  -fPIC -shared -xcce -DMEMORY_BASE
  -O2 -std=c++17 -Wno-ignored-attributes
  --cce-aicore-arch=dav-c220-vec
  "-I${PTO_ROOT}/include"
)

bisheng "${BFLAGS[@]}" \
  -DKERNEL_CPP=\"outputs/sinkhorn_batch8_generated.cpp\" \
  caller_sinkhorn_k4.cpp \
  -o outputs/kernel_sinkhorn.so

bisheng "${BFLAGS[@]}" \
  -DKERNEL_CPP=\"outputs/sinkhorn_k4_generated.cpp\" \
  caller_sinkhorn_k4.cpp \
  -o outputs/kernel_sinkhorn_naive.so

echo "Built outputs/kernel_sinkhorn.so (batched) and outputs/kernel_sinkhorn_naive.so (naive)."
