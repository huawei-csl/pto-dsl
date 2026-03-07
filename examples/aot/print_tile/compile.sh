
#!/usr/bin/env bash
set -e

PTO_DIR="$ASCEND_HOME_PATH/include/pto"
PTO_BACKUP="$ASCEND_HOME_PATH/include/pto_hidden"

# This runs on exit
restore() {
    if [ -d "$PTO_BACKUP" ]; then
        mv "$PTO_BACKUP" "$PTO_DIR"
    fi
}
trap restore EXIT

# For now we have to hide the CANN built-in headers, and use the cloned pto-isa's
# c.f. https://gitcode.com/cann/pto-isa/issues/149
mv "$PTO_DIR" "$PTO_BACKUP"

PTO_LIB_PATH=/sources/pto-isa
bisheng \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 -std=c++17 \
    -xcce -Xhost-start -Xhost-end \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -D_DEBUG --cce-enable-print \
    -I${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/runtime/runtime \
    -I${PTO_LIB_PATH}/include \
    -std=gnu++17 \
    ./caller.cpp \
    -o ./add_lib.so
