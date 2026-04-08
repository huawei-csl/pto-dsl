#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
MLIR_PATH="${SCRIPT_DIR}/c2v.mlir"
GENERATED_CPP="${ARTIFACT_DIR}/c2v.cpp"
LIB_PATH="${SCRIPT_DIR}/tpushpop_mlir_lib.so"

mkdir -p "${ARTIFACT_DIR}"
rm -f "${GENERATED_CPP}" "${LIB_PATH}"

MLIR_GEN_PATH="${SCRIPT_DIR}/c2v_gen.mlir"
python c2v_builder.py > c2v_gen.mlir
ptoas --pto-arch=a3 --enable-insert-sync "${MLIR_GEN_PATH}" > "${GENERATED_CPP}"


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
