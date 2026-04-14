set -e

rm -f \
    hadamard_auto_sync.pto hadamard_manual_sync.pto \
    hadamard_auto_sync.cpp hadamard_manual_sync.cpp \
    hadamard_auto_sync_lib.so hadamard_manual_sync_lib.so \
    hadamard_quant.pto hadamard_quant.cpp hadamard_quant_lib.so \
    hadamard_quant_manual_sync.pto hadamard_quant_manual_sync.cpp hadamard_quant_manual_sync_lib.so \
    hadamard_quant_gs*.pto hadamard_quant_gs*.cpp hadamard_quant_gs*_lib.so

# Auto-sync path: rely on ptoas synchronization insertion.
python ./hadamard_builder.py > ./hadamard_auto_sync.pto
ptoas --enable-insert-sync ./hadamard_auto_sync.pto -o ./hadamard_auto_sync.cpp

# Manual-sync path: explicit record/wait events from builder.
python ./hadamard_builder.py --manual-sync > ./hadamard_manual_sync.pto
ptoas ./hadamard_manual_sync.pto -o ./hadamard_manual_sync.cpp

# CANN 8.5 headers don't have CompactMode, need latest pto-isa source
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
    ./caller.cpp \
    -o ./hadamard_auto_sync_lib.so

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
    -DKERNEL_CPP="\"hadamard_manual_sync.cpp\"" \
    -DKERNEL_FN=fast_hadamard_manualsync \
    ./caller.cpp \
    -o ./hadamard_manual_sync_lib.so

# Fused Hadamard + quantize (fp16 → int8), uniform-quantization variant.
# Pass --group-size N to hadamard_quant_builder.py for a per-group variant.
python ./hadamard_quant_builder.py > ./hadamard_quant.pto
ptoas --enable-insert-sync ./hadamard_quant.pto -o ./hadamard_quant.cpp

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
    ./caller_quant.cpp \
    -o ./hadamard_quant_lib.so

python ./hadamard_quant_builder.py --manual-sync > ./hadamard_quant_manual_sync.pto
ptoas ./hadamard_quant_manual_sync.pto -o ./hadamard_quant_manual_sync.cpp

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
    -DKERNEL_CPP="\"hadamard_quant_manual_sync.cpp\"" \
    -DKERNEL_FN=fast_hadamard_quant_manualsync \
    ./caller_quant.cpp \
    -o ./hadamard_quant_manual_sync_lib.so

# Per-group quantization variant (group_size=128), auto-sync and manual-sync.
GROUP_SIZE=128

python ./hadamard_quant_builder.py --group-size ${GROUP_SIZE} > ./hadamard_quant_gs${GROUP_SIZE}.pto
ptoas --enable-insert-sync ./hadamard_quant_gs${GROUP_SIZE}.pto -o ./hadamard_quant_gs${GROUP_SIZE}.cpp

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
    -DKERNEL_CPP="\"hadamard_quant_gs${GROUP_SIZE}.cpp\"" \
    ./caller_quant.cpp \
    -o ./hadamard_quant_gs${GROUP_SIZE}_lib.so

python ./hadamard_quant_builder.py --group-size ${GROUP_SIZE} --manual-sync > ./hadamard_quant_gs${GROUP_SIZE}_manual_sync.pto
ptoas ./hadamard_quant_gs${GROUP_SIZE}_manual_sync.pto -o ./hadamard_quant_gs${GROUP_SIZE}_manual_sync.cpp

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
    -DKERNEL_CPP="\"hadamard_quant_gs${GROUP_SIZE}_manual_sync.cpp\"" \
    -DKERNEL_FN=fast_hadamard_quant_manualsync \
    ./caller_quant.cpp \
    -o ./hadamard_quant_gs${GROUP_SIZE}_manual_sync_lib.so
