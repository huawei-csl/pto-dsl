#!/usr/bin/env bash
set -e
rm -f sparse_attn.pto sparse_attn.cpp sparse_attn_lib.so

python ./sparse_attn_builder.py > ./sparse_attn.pto
ptoas --enable-insert-sync ./sparse_attn.pto -o ./sparse_attn.cpp

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
    -o ./sparse_attn_lib.so
