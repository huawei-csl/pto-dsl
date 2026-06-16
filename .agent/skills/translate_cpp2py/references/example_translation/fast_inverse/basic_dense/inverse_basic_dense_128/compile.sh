#!/usr/bin/env bash
set -e
python ./inverse_builder.py --matrix-size 128 > ./inverse_basic_dense_128.pto
ptoas --enable-insert-sync ./inverse_basic_dense_128.pto -o ./inverse_basic_dense_128.cpp
