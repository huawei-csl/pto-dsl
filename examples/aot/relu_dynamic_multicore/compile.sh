rm relu.pto generated_relu.cpp relu_lib.so
python relu_builder.py > ./relu.pto
ptoas --enable-insert-sync ./relu.pto > generated_relu.cpp

# CANN 8.5 heaer has no TRELU support, need latest source
PTO_LIB_PATH=/sources/pto-isa
bisheng -fPIC -shared -xcce \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -O2 -std=c++17 \
    -I${PTO_LIB_PATH}/include \
    ./caller.cpp \
    -o ./relu_lib.so
