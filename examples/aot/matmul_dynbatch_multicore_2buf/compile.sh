rm mul.cpp matmul_kernel.so

python ./matmul_dsl.py | ptoas > mul.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I${ASCEND_TOOLKIT_HOME}/include \
    --cce-soc-version=Ascend910B2 \
    --cce-soc-core-type=CubeCore \
    -I/mounted_home/pto-isa/include \
    ./caller.cpp \
    -o ./matmul_kernel.so
