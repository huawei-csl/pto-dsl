#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OP=${1:?Usage: compile.sh <op_name> [dtype]}
DTYPE=${2:-float32}

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

# Generate IR for both kernels
python "$SCRIPT_DIR/gen_ir.py" "$OP" 1d "$DTYPE" > "$TMP/${OP}_1d.pto"
ptoas --enable-insert-sync "$TMP/${OP}_1d.pto" -o "$TMP/${OP}_1d.cpp"

python "$SCRIPT_DIR/gen_ir.py" "$OP" 2d "$DTYPE" > "$TMP/${OP}_2d.pto"
ptoas --enable-insert-sync "$TMP/${OP}_2d.pto" -o "$TMP/${OP}_2d.cpp"

# Generate caller.cpp
python "$SCRIPT_DIR/caller.py" "$OP" "$DTYPE" > "$TMP/caller.cpp"

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
    -o "$SCRIPT_DIR/${OP}_${DTYPE}_lib.so"

echo "Built ${OP}_${DTYPE}_lib.so successfully."
