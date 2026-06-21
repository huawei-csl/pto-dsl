#!/usr/bin/env bash
set -e
rm -f hc_split_sinkhorn.pto hc_split_sinkhorn.cpp hc_split_sinkhorn_lib.so

python ./hc_split_sinkhorn_builder.py > ./hc_split_sinkhorn.pto
ptoas --enable-insert-sync ./hc_split_sinkhorn.pto -o ./hc_split_sinkhorn.cpp

PTO_LIB_PATH=${PTO_LIB_PATH:-/sources/pto-isa}
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
    -o ./hc_split_sinkhorn_lib.so
