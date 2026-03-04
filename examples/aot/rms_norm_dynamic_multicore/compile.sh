set -e

rm -f rms_norm.pto rms_norm.cpp rms_norm_lib.so

python ./rms_norm_builder.py > ./rms_norm.pto
ptoas --enable-insert-sync ./rms_norm.pto -o ./rms_norm.cpp

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
    -DKERNEL_CPP="\"rms_norm.cpp\"" \
    ./caller.cpp \
    -o ./rms_norm_lib.so
