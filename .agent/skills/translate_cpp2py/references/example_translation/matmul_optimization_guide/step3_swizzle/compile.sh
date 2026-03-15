#!/usr/bin/env bash
set -e
python ./step3_swizzle.py > ./step3_swizzle.pto
ptoas --enable-insert-sync ./step3_swizzle.pto -o ./step3_swizzle.cpp
