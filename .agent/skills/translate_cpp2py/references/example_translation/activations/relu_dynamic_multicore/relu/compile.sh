#!/usr/bin/env bash
set -e
python relu_builder.py > ./relu.pto
ptoas --enable-insert-sync ./relu.pto > generated_relu.cpp
