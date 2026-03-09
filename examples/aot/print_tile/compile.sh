
#!/usr/bin/env bash
set -e

PTO_DIR="$ASCEND_HOME_PATH/include/pto"
PTO_BACKUP="$ASCEND_HOME_PATH/include/pto_hidden"
PTO_LIB_PATH="/sources/pto-isa"
[ -d "$PTO_LIB_PATH" ] || exit 0


rm -f print_lib.so print_gen.cpp
python ./print_builder.py | ptoas --enable-insert-sync > print_gen.cpp

restore() {
    if [ -d "$PTO_BACKUP" ]; then
        mv "$PTO_BACKUP" "$PTO_DIR"
    fi
}

# For now we have to hide the CANN built-in headers, and use the cloned pto-isa's
# c.f. https://gitcode.com/cann/pto-isa/issues/149
mv "$PTO_DIR" "$PTO_BACKUP"

# Make restore run on EXIT
trap restore EXIT

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
    -o ./print_lib.so
