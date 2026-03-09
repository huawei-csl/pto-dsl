#!/usr/bin/env bash
set -euo pipefail

# Minimal reproducer for alloc_tile + addr level checks.

python ./ir_builder.py > noaddr.pto
python ./ir_builder.py --with-addr > withaddr.pto

echo "== Case A: default level, no addr (expected: success) =="
ptoas noaddr.pto -o noaddr.cpp

echo "== Case B: default level, with addr (expected: fail) =="
set +e
ptoas withaddr.pto -o withaddr_default.cpp
status_default=$?
set -e
echo "default-level with-addr exit code: ${status_default}"

echo "== Case C: level3, with addr (expected: success) =="
ptoas --pto-level=level3 withaddr.pto -o withaddr_level3.cpp

echo "== Case D: level3, no addr (expected: fail) =="
set +e
ptoas --pto-level=level3 noaddr.pto -o noaddr_level3.cpp
status_level3=$?
set -e
echo "level3 no-addr exit code: ${status_level3}"

echo "Done."

