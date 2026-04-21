#!/usr/bin/env bash
set -e
python ./matmul_builder.py > matmul.pto
ptoas --enable-insert-sync matmul.pto -o matmul.cpp
