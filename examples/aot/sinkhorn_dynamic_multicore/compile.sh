set -e

rm -f sinkhorn.pto sinkhorn.cpp sinkhorn_lib.so

python ./sinkhorn_builder.py > ./sinkhorn.pto
ptoas --enable-insert-sync ./sinkhorn.pto -o ./sinkhorn.cpp

# CANN 8.5 headers don't have CompactMode; need latest pto-isa source.
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
    -DKERNEL_CPP="\"sinkhorn.cpp\"" \
    ./caller.cpp \
    -o ./sinkhorn_lib.so
