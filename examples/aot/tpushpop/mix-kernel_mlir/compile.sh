#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
MODE="${TPUSHPOP_MODE:-c2v}"
BUILDER_PATH="${SCRIPT_DIR}/kernels/${MODE}_builder.py"
MLIR_GEN_PATH="${ARTIFACT_DIR}/${MODE}_gen.mlir"
GENERATED_CPP="${ARTIFACT_DIR}/${MODE}.cpp"
LIB_PATH="${ARTIFACT_DIR}/tpushpop_mlir_lib.so"


mkdir -p "${ARTIFACT_DIR}"
rm -f "${GENERATED_CPP}" "${LIB_PATH}"

python "${BUILDER_PATH}" > "${MLIR_GEN_PATH}"
ptoas --pto-arch=a3 --enable-insert-sync "${MLIR_GEN_PATH}" > "${GENERATED_CPP}"
# add extern "C" to function so kernel name is not mangled
perl -0pi -e 's/\b__global__ AICORE void call_both\(/extern "C" __global__ AICORE void call_both(/' "${GENERATED_CPP}"

bisheng \
    -I/sources/pto-isa/include/ \
    -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 -std=c++17 -g \
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong \
    -xcce -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -std=gnu++17 \
    -DKERNEL_CPP="\"${GENERATED_CPP}\"" \
    "${SCRIPT_DIR}/caller.cpp" \
    -o "${LIB_PATH}"

echo "Generated ${GENERATED_CPP}."
echo "Built ${LIB_PATH}."
