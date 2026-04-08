set -e

rm -f \
    hadamard_auto_sync.pto hadamard_manual_sync.pto \
    hadamard_auto_sync.cpp hadamard_manual_sync.cpp \
    hadamard_auto_sync_lib.so hadamard_manual_sync_lib.so

# Auto-sync path: rely on ptoas synchronization insertion.
python ./hadamard_builder.py > ./hadamard_auto_sync.pto
ptoas --enable-insert-sync ./hadamard_auto_sync.pto -o ./hadamard_auto_sync.cpp

# Manual-sync path: explicit record/wait events from builder.
python ./hadamard_builder.py --manual-sync > ./hadamard_manual_sync.pto
ptoas ./hadamard_manual_sync.pto -o ./hadamard_manual_sync.cpp

# CANN 8.5 headers don't have CompactMode, need latest pto-isa source
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
