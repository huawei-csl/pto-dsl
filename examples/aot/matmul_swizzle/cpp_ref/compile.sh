bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I${ASCEND_TOOLKIT_HOME}/include \
    ./matmul.cpp \
    -o ./matmul_kernel.so
