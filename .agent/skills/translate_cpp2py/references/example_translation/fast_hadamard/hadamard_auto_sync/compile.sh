#!/usr/bin/env bash
set -e
python ./hadamard_builder.py > ./hadamard_auto_sync.pto
ptoas --enable-insert-sync ./hadamard_auto_sync.pto -o ./hadamard_auto_sync.cpp
