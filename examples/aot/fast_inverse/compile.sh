#!/usr/bin/env bash
set -euo pipefail

rm -f \
    inverse_auto_sync_*.pto inverse_manual_sync_*.pto \
    inverse_auto_sync_*.cpp inverse_manual_sync_*.cpp \
    inverse_auto_sync_lib.so inverse_manual_sync_lib.so

SIZES=(16 32 64 96 128)

# Auto-sync path: rely on ptoas synchronization insertion.
for size in "${SIZES[@]}"; do
    python ./inverse_builder.py \
        --matrix-size "${size}" \
        --kernel-name "tri_inv_trick_fp16_${size}" \
        > "./inverse_auto_sync_${size}.pto"
    ptoas --enable-insert-sync "./inverse_auto_sync_${size}.pto" -o "./inverse_auto_sync_${size}.cpp"
done

# Manual-sync path: explicit record/wait events from builder.
for size in "${SIZES[@]}"; do
    python ./inverse_builder.py \
        --manual-sync \
        --matrix-size "${size}" \
        --kernel-name "tri_inv_trick_fp16_${size}" \
        > "./inverse_manual_sync_${size}.pto"
    ptoas "./inverse_manual_sync_${size}.pto" -o "./inverse_manual_sync_${size}.cpp"
done

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
    ./caller.cpp \
    -o ./inverse_auto_sync_lib.so

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
    -DKERNEL_CPP_16="\"inverse_manual_sync_16.cpp\"" \
    -DKERNEL_CPP_32="\"inverse_manual_sync_32.cpp\"" \
    -DKERNEL_CPP_64="\"inverse_manual_sync_64.cpp\"" \
    -DKERNEL_CPP_96="\"inverse_manual_sync_96.cpp\"" \
    -DKERNEL_CPP_128="\"inverse_manual_sync_128.cpp\"" \
    ./caller.cpp \
    -o ./inverse_manual_sync_lib.so
