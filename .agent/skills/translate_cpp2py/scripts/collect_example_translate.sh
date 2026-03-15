#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
AOT_DIR="${REPO_ROOT}/examples/aot"
OUT_DIR="${AOT_DIR}/example_translation"

if [[ ! -d "${AOT_DIR}" ]]; then
  echo "AOT examples directory not found: ${AOT_DIR}" >&2
  exit 1
fi

export AOT_DIR OUT_DIR

python3 "${SCRIPT_DIR}/collect_example_translate.py"
