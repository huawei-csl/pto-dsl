#!/usr/bin/env bash
set -e
python ./add_double_builder.py > ./add_double.pto
ptoas --enable-insert-sync ./add_double.pto -o ./add_double.cpp
