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
    local SRC=$1
    local DST=$2
    local RMODE=${3:-}

    local STEM="cvt_${SRC}_to_${DST}"
    local RMODE_ARG=""
    if [ -n "$RMODE" ]; then
        STEM="${STEM}_${RMODE}"
        RMODE_ARG="--rmode $RMODE"
    fi

    python "$SCRIPT_DIR/gen_ir.py" --src-dtype "$SRC" --dst-dtype "$DST" $RMODE_ARG \
        > "$TMP/${STEM}.pto"
    ptoas --enable-insert-sync "$TMP/${STEM}.pto" -o "$TMP/${STEM}.cpp"
    python "$SCRIPT_DIR/caller.py" --src-dtype "$SRC" --dst-dtype "$DST" $RMODE_ARG \
        > "$TMP/caller_${STEM}.cpp"
    _bisheng "$TMP/caller_${STEM}.cpp" -o "$SCRIPT_DIR/${STEM}_lib.so"
    echo "Built ${STEM}_lib.so"
}

# Three conversions covering float narrowing, float→int (with int8), and int→int
# (matches _CONVERSIONS in test_cvt.py)

# float narrowing — floor
compile_kernel float32 float16 floor

# float16 → small signed int — cast_rint
compile_kernel float16 int8 cast_rint

# int → int narrowing
compile_kernel int32 int16
