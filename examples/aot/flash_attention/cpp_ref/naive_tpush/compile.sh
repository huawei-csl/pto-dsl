#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
NPU_ARCH="${NPU_ARCH:-dav-2201}"

MLIR_PATH="${ARTIFACT_DIR}/fa_dsl.mlir"
GENERATED_CPP="${ARTIFACT_DIR}/fa_dsl.cpp"
LIB_PATH="${ARTIFACT_DIR}/fa_dsl.so"
FA_DSL_BUILDER="${FA_DSL_BUILDER:-fa_dsl_builder.py}"
BUILDER_PATH="${SCRIPT_DIR}/kernels/${FA_DSL_BUILDER}"
PTOAS_SYNC_ARGS=(--enable-insert-sync)

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [--manual-sync]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    case "$1" in
        --manual-sync)
            PTOAS_SYNC_ARGS=()
            ;;
        *)
            echo "Usage: $0 [--manual-sync]" >&2
            exit 2
            ;;
    esac
fi

mkdir -p "${ARTIFACT_DIR}"
rm -f "${MLIR_PATH}" "${GENERATED_CPP}" "${LIB_PATH}"

if [[ ! -f "${BUILDER_PATH}" ]]; then
    echo "Builder not found: ${BUILDER_PATH}" >&2
    exit 2
fi

python "${BUILDER_PATH}" > "${MLIR_PATH}"
ptoas --pto-arch=a3 "${PTOAS_SYNC_ARGS[@]}" "${MLIR_PATH}" > "${GENERATED_CPP}"

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
    --npu-arch="${NPU_ARCH}" -DMEMORY_BASE \
    -std=gnu++17 \
    -DKERNEL_CPP="\"${GENERATED_CPP}\"" \
    "${SCRIPT_DIR}/caller.cpp" \
    -o "${LIB_PATH}"

echo "Generated ${GENERATED_CPP}."
echo "Built ${LIB_PATH}."
