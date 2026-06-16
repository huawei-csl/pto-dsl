set -e

rm -f hardswish.pto hardswish.cpp hardswish_lib.so

python ./hardswish_builder.py > ./hardswish.pto
ptoas --enable-insert-sync ./hardswish.pto -o ./hardswish.cpp

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
    -DKERNEL_CPP="\"hardswish.cpp\"" \
    ./caller.cpp \
    -o ./hardswish_lib.so

