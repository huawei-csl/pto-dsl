#!/usr/bin/env bash
set -euo pipefail

# Minimal reproducer for alloc_tile + addr level checks.

python ./ir_builder.py > noaddr.pto
python ./ir_builder.py --with-addr > withaddr.pto

echo "== Case A: default level, no addr (expected: success) =="
ptoas noaddr.pto -o noaddr.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    ./noaddr.cpp \
    -o ./noaddr.so

echo "== Case B: level3, with addr (expected: success at ptoas, but fail at bisheng) =="
ptoas --pto-level=level3 withaddr.pto -o withaddr_level3.cpp

bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -I"${ASCEND_TOOLKIT_HOME}/include" \
    ./withaddr_level3.cpp \
    -o ./withaddr_level3.so

echo "Done."

