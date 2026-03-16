#!/usr/bin/env bash
set -euo pipefail

rm -f \
    inverse_auto_sync.pto inverse_manual_sync.pto \
    inverse_auto_sync.cpp inverse_manual_sync.cpp \
    inverse_auto_sync_lib.so inverse_manual_sync_lib.so

# Auto-sync path: rely on ptoas synchronization insertion.
python ./inverse_builder.py > ./inverse_auto_sync.pto
ptoas --enable-insert-sync ./inverse_auto_sync.pto -o ./inverse_auto_sync.cpp

# Manual-sync path: explicit record/wait events from builder.
python ./inverse_builder.py --manual-sync > ./inverse_manual_sync.pto
ptoas ./inverse_manual_sync.pto -o ./inverse_manual_sync.cpp

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
    -DKERNEL_CPP="\"inverse_manual_sync.cpp\"" \
    ./caller.cpp \
    -o ./inverse_manual_sync_lib.so
