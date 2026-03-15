#!/usr/bin/env bash
set -euo pipefail

rm -f repro_fail_tmov_acc_to_mat.pto

echo "[expect-fail] Emitting PTO from failing reproducer..."
if python3 ./repro_fail_tmov_acc_to_mat.py > ./repro_fail_tmov_acc_to_mat.pto; then
  echo "Unexpected success: reproducer should fail during IR verification."
  exit 1
else
  echo "Expected failure observed."
fi
