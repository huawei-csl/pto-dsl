#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PTO_LIB_PATH=${PTO_LIB_PATH:-/sources/pto-isa}

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

_bisheng() {
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
        "$@"
}

compile_kernel() {
    local VARIANT=$1
    local STEM="quant_${VARIANT}_dynamic"

    python "$SCRIPT_DIR/gen_ir.py" "$VARIANT" > "$TMP/${STEM}.pto"
    ptoas --enable-insert-sync "$TMP/${STEM}.pto" -o "$TMP/${STEM}.cpp"
    python "$SCRIPT_DIR/caller.py" "$VARIANT" > "$TMP/caller_${STEM}.cpp"
    _bisheng "$TMP/caller_${STEM}.cpp" -o "$SCRIPT_DIR/${STEM}_lib.so"
    echo "Built ${STEM}_lib.so"
}

compile_kernel sym
compile_kernel asym
