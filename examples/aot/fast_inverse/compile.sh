set -e

rm -f \
    tri_inv_trick_auto_sync.pto tri_inv_trick_manual_sync.pto \
    tri_inv_trick_auto_sync.cpp tri_inv_trick_manual_sync.cpp \
    tri_inv_trick_auto_sync_lib.so tri_inv_trick_manual_sync_lib.so

# Auto-sync path: rely on ptoas synchronization insertion.
python ./inverse_builder.py > ./tri_inv_trick_auto_sync.pto
ptoas --enable-insert-sync ./tri_inv_trick_auto_sync.pto -o ./tri_inv_trick_auto_sync.cpp

# Manual-sync path: explicit record/wait events from builder.
python ./inverse_builder.py --manual-sync > ./tri_inv_trick_manual_sync.pto
ptoas ./tri_inv_trick_manual_sync.pto -o ./tri_inv_trick_manual_sync.cpp

PTO_LIB_PATH=/sources/pto-isa
# PTO_LIB_PATH=$ASCEND_TOOLKIT_HOME

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
    -o ./tri_inv_trick_auto_sync_lib.so

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
    -DKERNEL_CPP="\"tri_inv_trick_manual_sync.cpp\"" \
    -DKERNEL_FN=tri_inv_trick_fp16_manualsync \
    ./caller.cpp \
    -o ./tri_inv_trick_manual_sync_lib.so
