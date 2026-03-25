#!/bin/bash
# Compile one TopK kernel config into a shared library.
#
# Usage: bash compile.sh [N_COLS] [TOPK] [BLOCK_DIM]
# Defaults: 512 256 24
#
# N_ROWS is a runtime argument – the same library handles any row count.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_COLS=${1:-512}
TOPK=${2:-256}
BLOCK_DIM=${3:-24}

FN="topk_c${N_COLS}_k${TOPK}"

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

python "$SCRIPT_DIR/topk_builder.py" \
    --n-cols "$N_COLS" --topk "$TOPK" --block-dim "$BLOCK_DIM" \
    > "$TMP/${FN}.pto"
ptoas --enable-insert-sync "$TMP/${FN}.pto" -o "$TMP/${FN}.cpp"

python "$SCRIPT_DIR/caller.py" "$FN" --block-dim "$BLOCK_DIM" > "$TMP/caller.cpp"

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
    -o "$SCRIPT_DIR/${FN}_lib.so"

echo "Built ${FN}_lib.so successfully."
