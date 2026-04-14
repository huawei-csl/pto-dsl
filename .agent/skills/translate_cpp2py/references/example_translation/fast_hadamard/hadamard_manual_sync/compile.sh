#!/usr/bin/env bash
set -e
python ./hadamard_builder.py --manual-sync > ./hadamard_manual_sync.pto
ptoas ./hadamard_manual_sync.pto -o ./hadamard_manual_sync.cpp
