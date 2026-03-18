#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="./build_artifacts"
MATRIX_SIZE="${1:-64}"
VARIANT="${2:-single}"

if [[ "${VARIANT}" != "single" && "${VARIANT}" != "double" ]]; then
    echo "Usage: bash compile.sh [matrix_size] [single|double]"
    exit 1
fi

SUFFIX=""
BUILDER_ARGS=(--matrix-size "${MATRIX_SIZE}")
if [[ "${VARIANT}" == "double" ]]; then
    SUFFIX="_db"
    BUILDER_ARGS+=(--double-buffer)
fi

mkdir -p "${ARTIFACT_DIR}"
rm -f "${ARTIFACT_DIR}/inverse${SUFFIX}.pto" "${ARTIFACT_DIR}/inverse${SUFFIX}.cpp" "inverse_lib${SUFFIX}.so"

python ./inverse_builder.py "${BUILDER_ARGS[@]}" > "${ARTIFACT_DIR}/inverse${SUFFIX}.pto"

ptoas --enable-insert-sync "${ARTIFACT_DIR}/inverse${SUFFIX}.pto" -o "${ARTIFACT_DIR}/inverse${SUFFIX}.cpp"

PTO_LIB_PATH=/sources/pto-isa
# PTO_LIB_PATH=$ASCEND_TOOLKIT_HOME

bisheng \
    -I${PTO_LIB_PATH}/include \
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
    -DKERNEL_CPP="\"${ARTIFACT_DIR}/inverse${SUFFIX}.cpp\"" \
    ./caller.cpp \
    -o "./inverse_lib${SUFFIX}.so"
