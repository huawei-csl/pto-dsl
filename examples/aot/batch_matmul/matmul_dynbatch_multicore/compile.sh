rm matmul.pto matmul.cpp matmul_kernel.so

python ./matmul_builder.py > matmul.pto
ptoas matmul.pto -o matmul.cpp

# CANN 8.5 headers don't have CompactMode, need latest pto-isa source
PTO_LIB_PATH=/sources/pto-isa
bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I${PTO_LIB_PATH}/include \
    ./caller.cpp \
    -o ./matmul_kernel.so
