#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DTYPE=${1:?Usage: compile.sh <dtype> <n>}
N=${2:?Usage: compile.sh <dtype> <n>}

CASE_ID="${DTYPE}_n${N}"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# Generate IR and compile the hadamard kernel
python "$SCRIPT_DIR/gen_ir.py" "$DTYPE" "$N" > "$SCRIPT_DIR/${CASE_ID}.pto"
ptoas --enable-insert-sync "$SCRIPT_DIR/${CASE_ID}.pto" -o "$SCRIPT_DIR/${CASE_ID}.cpp"

# Generate caller.cpp
python "$SCRIPT_DIR/caller.py" "$DTYPE" "$N" > "$SCRIPT_DIR/caller.cpp"

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
    "$SCRIPT_DIR/caller.cpp" \
    -o "$SCRIPT_DIR/${CASE_ID}_lib.so"


echo "Built ${CASE_ID}_lib.so successfully."
