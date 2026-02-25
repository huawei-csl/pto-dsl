rm -f add.pto add.cpp add_lib.so
rm -f add_double.pto add_double.cpp add_double_lib.so

build_variant() {
    local builder_path="$1"
    local pto_path="$2"
    local cpp_path="$3"
    local lib_path="$4"

    python "${builder_path}" > "${pto_path}"
    ptoas --enable-insert-sync "${pto_path}" -o "${cpp_path}"

    bisheng \
        -I${ASCEND_TOOLKIT_HOME}/include \
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
        -DKERNEL_CPP="\"${cpp_path}\"" \
        ./caller.cpp \
        -o "${lib_path}"
}

build_variant "./add_builder.py" "./add.pto" "add.cpp" "./add_lib.so"
build_variant "./add_double_builder.py" "./add_double.pto" "add_double.cpp" "./add_double_lib.so"
