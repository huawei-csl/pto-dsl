#!/usr/bin/env bash
set -euo pipefail

rm -f repro_manual_acc_to_mat.so

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
  --npu-arch=dav-2201 -DMEMORY_BASE \
  -I"${ASCEND_TOOLKIT_HOME}/include" \
  ./repro_manual_acc_to_mat.cpp \
  -o ./repro_manual_acc_to_mat.so

echo "Generated: repro_manual_acc_to_mat.so"
