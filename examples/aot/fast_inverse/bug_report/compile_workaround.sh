#!/usr/bin/env bash
set -euo pipefail

rm -f repro_workaround_spill_acc_to_mat.pto repro_workaround_spill_acc_to_mat.cpp

python3 ./repro_workaround_spill_acc_to_mat.py > ./repro_workaround_spill_acc_to_mat.pto
ptoas --enable-insert-sync ./repro_workaround_spill_acc_to_mat.pto -o ./repro_workaround_spill_acc_to_mat.cpp

echo "Generated:"
echo "  - repro_workaround_spill_acc_to_mat.pto"
echo "  - repro_workaround_spill_acc_to_mat.cpp"
