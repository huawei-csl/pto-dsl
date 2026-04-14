#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TMP=$(mktemp -d)
trap "rm -rf \"$TMP\"" EXIT


PTO_LIB_PATH=${PTO_LIB_PATH:-/sources/pto-isa}
BISHENG_FLAGS=(
    -I${PTO_LIB_PATH}/include
    -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong
    -xcce -Xhost-start -Xhost-end
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    --npu-arch=dav-2201 -DMEMORY_BASE
    -std=gnu++17
)

MODES=(
    colexpand
    colexpand_sub
    colexpand_div
    colexpand_mul
    colexpand_min
    colexpand_max
    colexpand_add
    colexpand_expdif
    rowexpand
    rowexpand_add
    rowexpand_mul
    rowexpand_sub
    rowexpand_div
    rowexpand_min
    rowexpand_max
    rowexpand_expdif
)

for MODE in "${MODES[@]}"; do
    python "$SCRIPT_DIR/gen_ir.py" --mode "$MODE" > "$TMP/${MODE}.pto"
    ptoas --enable-insert-sync "$TMP/${MODE}.pto" -o "$TMP/${MODE}.cpp"

    python "$SCRIPT_DIR/caller.py" --mode "$MODE" > "$TMP/${MODE}_caller.cpp"

    bisheng "${BISHENG_FLAGS[@]}" \
        "$TMP/${MODE}_caller.cpp" \
        -o "$SCRIPT_DIR/${MODE}_lib.so"

    echo "Built ${MODE}_lib.so successfully."
done
