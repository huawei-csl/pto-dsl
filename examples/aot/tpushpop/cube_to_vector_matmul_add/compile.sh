#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
LIB_PATH="${SCRIPT_DIR}/tpushpop_cv_lib.so"
EXTRA_BISHENG_FLAGS="${EXTRA_BISHENG_FLAGS:-}"

if [[ "${TPUSHPOP_SANITY_ONLY:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    EXTRA_BISHENG_FLAGS="${EXTRA_BISHENG_FLAGS} -DTPUSHPOP_SANITY_ONLY"
fi

PTO_INCLUDE_PATH="${PTO_INCLUDE_PATH:-/sources/pto-isa/include/}"
if [[ ! -d "${PTO_INCLUDE_PATH}" ]]; then
    if [[ -n "${PTO_LIB_PATH:-}" && -d "${PTO_LIB_PATH}/include" ]]; then
        PTO_INCLUDE_PATH="${PTO_LIB_PATH}/include"
    elif [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -d "${ASCEND_TOOLKIT_HOME}/include" ]]; then
        PTO_INCLUDE_PATH="${ASCEND_TOOLKIT_HOME}/include"
    else
        echo "Could not find PTO headers. Set PTO_INCLUDE_PATH, PTO_LIB_PATH, or ASCEND_TOOLKIT_HOME." >&2
        exit 1
    fi
fi

mkdir -p "${ARTIFACT_DIR}"
rm -f "${LIB_PATH}"

bisheng \
    -I"${PTO_INCLUDE_PATH}" \
    -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 -std=c++17 \
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong \
    -xcce -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -std=gnu++17 \
    ${EXTRA_BISHENG_FLAGS} \
    -DKERNEL_CPP="\"${REPO_ROOT}/refs/tpushpop_cv.cpp\"" \
    "${SCRIPT_DIR}/caller.cpp" \
    -o "${LIB_PATH}"

echo "Built ${LIB_PATH}."
