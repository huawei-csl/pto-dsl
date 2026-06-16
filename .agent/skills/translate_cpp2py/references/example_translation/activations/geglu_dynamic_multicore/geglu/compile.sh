#!/usr/bin/env bash
set -e
python ./geglu_builder.py > ./geglu.pto
ptoas --enable-insert-sync ./geglu.pto -o ./geglu.cpp
