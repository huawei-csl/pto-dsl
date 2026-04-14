#!/usr/bin/env bash
set -e
python ./step2_doublebuffer.py > ./step2_doublebuffer.pto
ptoas --enable-insert-sync ./step2_doublebuffer.pto -o ./step2_doublebuffer.cpp
