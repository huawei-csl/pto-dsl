#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DTYPE=${1:?Usage: compile.sh <dtype> [mask_pattern]}
MASK=${2:-P1111}

FN_NAME="vec_gather_2d_dynamic_${DTYPE}_${MASK}"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# Generate IR and compile the gather kernel
python "$SCRIPT_DIR/gen_ir.py" "$DTYPE" "$MASK" > "$TMP/${FN_NAME}.pto"
ptoas --enable-insert-sync "$TMP/${FN_NAME}.pto" -o "$TMP/${FN_NAME}.cpp"

# Generate caller.cpp
python "$SCRIPT_DIR/caller.py" "$DTYPE" "$MASK" > "$TMP/caller.cpp"

PTO_LIB_PATH=/sources/pto-isa
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
    "$TMP/caller.cpp" \
    -o "$SCRIPT_DIR/${FN_NAME}_lib.so"

echo "Built ${FN_NAME}_lib.so successfully."
