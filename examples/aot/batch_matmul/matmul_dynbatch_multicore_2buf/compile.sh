rm mul.cpp matmul_kernel.so

python ./matmul_dsl.py | ptoas > mul.cpp

# CANN 8.5 headers don't have CompactMode, need latest pto-isa source
PTO_LIB_PATH=/sources/pto-isa
bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I${PTO_LIB_PATH}/include \
    --cce-soc-version=Ascend910B2 \
    --cce-soc-core-type=CubeCore \
    ./caller.cpp \
    -o ./matmul_kernel.so
