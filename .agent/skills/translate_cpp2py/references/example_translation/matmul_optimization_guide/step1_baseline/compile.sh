#!/usr/bin/env bash
set -e
python ./step1_baseline.py > ./step1_baseline.pto
ptoas --enable-insert-sync ./step1_baseline.pto -o ./step1_baseline.cpp
