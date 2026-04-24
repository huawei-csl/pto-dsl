#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
#
# AOT-compile the flash-attention kernel for one or more sequence lengths.
#
# Usage:
#   bash compile.sh                    # build the default set: NUM_TILES = 16,32,64,128
#                                      # -> fa.so, fa_32.so, fa_64.so, fa_128.so
#                                      # (NUM_TILES=16 → 8k seqlen → fa.so)
#   FA_TILES=16,64 bash compile.sh     # build only the listed NUM_TILES variants
#   FA_TILES=16    bash compile.sh     # single-variant build (legacy behavior)
#
# Each NUM_TILES value N produces fa${TAG}.{mlir,cpp,so} where
#   TAG = ""    if N == 16   (the builder's default → plain "fa.so")
#   TAG = "_N"  otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"

mkdir -p "${ARTIFACT_DIR}"

build_variant() {
    local num_tiles="$1"
    local tag
    if [[ "${num_tiles}" == "16" ]]; then
        tag=""
    else
        tag="_${num_tiles}"
    fi
    local mlir_path="${ARTIFACT_DIR}/fa${tag}.mlir"
    local generated_cpp="${ARTIFACT_DIR}/fa${tag}.cpp"
    local lib_path="${ARTIFACT_DIR}/fa${tag}.so"

    echo "==> Building NUM_TILES=${num_tiles} -> $(basename "${lib_path}")"
    rm -f "${mlir_path}" "${generated_cpp}" "${lib_path}"

    FA_NUM_TILES="${num_tiles}" \
        python "${SCRIPT_DIR}/kernels/fa_builder.py" > "${mlir_path}"
    ptoas --pto-arch=a3 --enable-insert-sync "${mlir_path}" > "${generated_cpp}"

    bisheng \
        -I"${PTO_LIB_PATH}/include" \
        -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 -std=c++17 \
        -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong \
        -xcce -Xhost-start -Xhost-end \
        -mllvm -cce-aicore-stack-size=0x8000 \
        -mllvm -cce-aicore-function-stack-size=0x8000 \
        -mllvm -cce-aicore-record-overflow=true \
        -mllvm -cce-aicore-addr-transform \
        -mllvm -cce-aicore-dcci-insert-for-scalar=false \
        -cce-enable-mix \
        --npu-arch=dav-2201 -DMEMORY_BASE \
        -std=gnu++17 \
        -DKERNEL_CPP="\"${generated_cpp}\"" \
        "${SCRIPT_DIR}/caller.cpp" \
        -o "${lib_path}"

    echo "    built ${lib_path}"
}

# Default tile set covers seqlen = NUM_TILES * S1_TILE = NUM_TILES * 512
#   16 -> 8k, 32 -> 16k, 64 -> 32k, 128 -> 64k
FA_TILES="${FA_TILES:-16,32,64,128}"

IFS=',' read -r -a tile_list <<< "${FA_TILES}"
for nt in "${tile_list[@]}"; do
    nt_trim="$(echo "${nt}" | tr -d '[:space:]')"
    [[ -z "${nt_trim}" ]] && continue
    build_variant "${nt_trim}"
done

echo "Done. Built variants: ${FA_TILES}"
