#!/usr/bin/env bash
set -e
python ./matmul_builder.py > matmul.pto
ptoas matmul.pto -o matmul.cpp
