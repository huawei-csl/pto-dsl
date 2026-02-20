rm matmul.pto matmul.cpp matmul_kernel.so

python ./matmul_builder.py > matmul.pto
ptoas matmul.pto -o matmul.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I${ASCEND_TOOLKIT_HOME}/include \
    ./caller.cpp \
    -o ./matmul_kernel.so
