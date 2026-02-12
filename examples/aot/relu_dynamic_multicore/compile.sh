rm relu.pto generated_relu.cpp relu_lib.so
python relu_builder.py > ./relu.pto
ptoas --enable-insert-sync ./relu.pto > generated_relu.cpp

bisheng -fPIC -shared -xcce \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -O2 -std=c++17 \
    -I${ASCEND_TOOLKIT_HOME}/include \
    ./caller.cpp \
    -o ./relu_lib.so


# bisheng \
#     -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 -std=c++17 \
#     -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong \
#     -xcce -Xhost-start -Xhost-end \
#     -mllvm -cce-aicore-stack-size=0x8000 \
#     -mllvm -cce-aicore-function-stack-size=0x8000 \
#     -mllvm -cce-aicore-record-overflow=true \
#     -mllvm -cce-aicore-addr-transform \
#     -mllvm -cce-aicore-dcci-insert-for-scalar=false \
#     --npu-arch=dav-2201 -DMEMORY_BASE \
#     -std=gnu++17 \
#     ./caller.cpp \
#     -o ./relu_lib.so
