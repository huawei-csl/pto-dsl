#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
MLIR_PATH="${ARTIFACT_DIR}/fa.mlir"
GENERATED_CPP="${ARTIFACT_DIR}/fa.cpp"
LIB_PATH="${ARTIFACT_DIR}/fa.so"

PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"

mkdir -p "${ARTIFACT_DIR}"
rm -f "${GENERATED_CPP}" "${LIB_PATH}"

python "${SCRIPT_DIR}/kernels/fa_builder.py" > "${MLIR_PATH}"
ptoas --pto-arch=a3 --enable-insert-sync "${MLIR_PATH}" > "${GENERATED_CPP}"
# Per-block GM-slot partitioning is done in the DSL via pto.add_ptr in
# call_both; no post-processing of the generated C++ needed.

bisheng \
    -I"${PTO_LIB_PATH}/include" \
    -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 -std=c++17 \
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong \
    -xcce -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -cce-enable-mix \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -std=gnu++17 \
    -DKERNEL_CPP="\"${GENERATED_CPP}\"" \
    "${SCRIPT_DIR}/caller.cpp" \
    -o "${LIB_PATH}"

echo "Generated ${GENERATED_CPP}."
echo "Built ${LIB_PATH}."
