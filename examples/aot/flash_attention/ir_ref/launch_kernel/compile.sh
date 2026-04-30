#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
#
# AOT-compile ../fa.cpp (ptoas output from fa.pto; regenerate via ../gen_cpp.sh) into host-loaded fa.so.
# Geometry baked into IR: Q_ROWS=2048, S1_TOTAL=4096 (NUM_TILES=16 × S1_TILE=256), HEAD=128.
#
# Usage:
#   bash compile.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
KERNEL_CPP="${SCRIPT_DIR}/../fa.cpp"
GENERATED_SO="${ARTIFACT_DIR}/fa.so"

mkdir -p "${ARTIFACT_DIR}"
rm -f "${GENERATED_SO}"

echo "==> bisheng ../fa.cpp -> ${GENERATED_SO}"
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
    -DKERNEL_CPP="\"${KERNEL_CPP}\"" \
    "${SCRIPT_DIR}/caller.cpp" \
    -o "${GENERATED_SO}"

{
    echo "FA_NUM_TILES=16"
    echo "FA_S1_TILE=256"
    echo "FA_Q_ROWS=2048"
} >"${ARTIFACT_DIR}/fa.build_env"

echo "Done."
