#!/usr/bin/env bash
set -e
python ./add_builder.py > ./add.pto
ptoas --enable-insert-sync ./add.pto -o ./add.cpp
