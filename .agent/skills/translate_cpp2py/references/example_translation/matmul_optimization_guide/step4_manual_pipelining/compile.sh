#!/usr/bin/env bash
set -e
python ./step4_manual_pipelining.py > ./step4_manual_pipelining.pto
ptoas ./step4_manual_pipelining.pto -o ./step4_manual_pipelining.cpp
