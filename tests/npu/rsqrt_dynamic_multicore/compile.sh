#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DTYPE=${1:-float32}

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

python "$SCRIPT_DIR/gen_ir.py" "$DTYPE" > "$TMP/rsqrt_${DTYPE}.pto"
ptoas --enable-insert-sync "$TMP/rsqrt_${DTYPE}.pto" -o "$TMP/rsqrt_${DTYPE}.cpp"

python "$SCRIPT_DIR/caller.py" "$DTYPE" > "$TMP/caller.cpp"

bisheng \
    -I${ASCEND_TOOLKIT_HOME}/include \
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
    -o "$SCRIPT_DIR/rsqrt_${DTYPE}_lib.so"

echo "Built rsqrt_${DTYPE}_lib.so successfully."
