#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/compile_common.sh"

ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
NPU_ARCH="${NPU_ARCH:-dav-2201}"
PTO_LEVEL="${PTO_LEVEL:-}"

MLIR_PATH="${ARTIFACT_DIR}/fa_dsl.mlir"
GENERATED_CPP="${ARTIFACT_DIR}/fa_dsl.cpp"
PATCHED_CPP="${ARTIFACT_DIR}/fa_dsl_patched.cpp"
LIB_PATH="${ARTIFACT_DIR}/fa_dsl.so"
RUNTIME_BUILDER_PATH="${ARTIFACT_DIR}/fa_dsl_runtime_builder.py"
BUILDER_PATH="${SCRIPT_DIR}/fa_dsl_builder_tile512.py"

parse_common_compile_args "$@"

mkdir -p "${ARTIFACT_DIR}"
rm -f "${MLIR_PATH}" "${GENERATED_CPP}" "${PATCHED_CPP}" "${LIB_PATH}" "${RUNTIME_BUILDER_PATH}"

python "${BUILDER_PATH}" > "${MLIR_PATH}"

PTOAS_ARGS=(--pto-arch=a3)
if [[ -n "${PTO_LEVEL}" ]]; then
    PTOAS_ARGS+=("--pto-level=${PTO_LEVEL}")
fi
PTOAS_ARGS+=("${PTOAS_SYNC_ARGS[@]}")

ptoas "${PTOAS_ARGS[@]}" "${MLIR_PATH}" > "${GENERATED_CPP}"
maybe_patch_vec_barriers "${GENERATED_CPP}" "${PATCHED_CPP}" "${REMOVE_VEC_BARRIER_LINES}"

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
    -DKERNEL_CPP="\"${PATCHED_CPP}\"" \
    "${SCRIPT_DIR}/caller.cpp" \
    -o "${LIB_PATH}"

cp "${BUILDER_PATH}" "${RUNTIME_BUILDER_PATH}"

echo "Generated ${GENERATED_CPP}."
echo "Built ${LIB_PATH}."
echo "Runtime builder ${RUNTIME_BUILDER_PATH}."
