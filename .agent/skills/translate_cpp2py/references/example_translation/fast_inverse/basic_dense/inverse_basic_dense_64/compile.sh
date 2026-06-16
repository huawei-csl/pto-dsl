#!/usr/bin/env bash
set -e
python ./inverse_builder.py --matrix-size 64 > ./inverse_basic_dense_64.pto
ptoas --enable-insert-sync ./inverse_basic_dense_64.pto -o ./inverse_basic_dense_64.cpp
