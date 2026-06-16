#!/usr/bin/env bash
set -e
python ./inverse_builder.py --matrix-size 64 > ./inverse_block_inversion_64.pto
ptoas --enable-insert-sync ./inverse_block_inversion_64.pto -o ./inverse_block_inversion_64.cpp
