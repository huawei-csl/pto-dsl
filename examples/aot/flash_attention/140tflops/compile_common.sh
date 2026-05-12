#!/usr/bin/env bash

parse_common_compile_args() {
    PTOAS_SYNC_ARGS=(--enable-insert-sync)
    REMOVE_VEC_BARRIER_LINES=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --remove-vec-barriers)
                if [[ $# -lt 2 || -z "$2" ]]; then
                    echo "--remove-vec-barriers requires a comma-separated line list" >&2
                    exit 2
                fi
                REMOVE_VEC_BARRIER_LINES="$2"
                shift 2
                ;;
            *)
                echo "Usage: $0 [--remove-vec-barriers line1,line2,...]" >&2
                exit 2
                ;;
        esac
    done
}

maybe_patch_vec_barriers() {
    local src_cpp="$1"
    local dst_cpp="$2"
    local raw_lines="$3"

    if [[ -z "${raw_lines}" ]]; then
        PATCHED_CPP="${src_cpp}"
        return
    fi

    python - "${src_cpp}" "${dst_cpp}" "${raw_lines}" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
remove_lines = {int(part.strip()) for part in sys.argv[3].split(",") if part.strip()}

lines = src.read_text().splitlines()
patched = []
for i, line in enumerate(lines, start=1):
    if i in remove_lines and "pipe_barrier(PIPE_V);" in line:
        patched.append("  /* removed PIPE_V barrier via --remove-vec-barriers */")
    else:
        patched.append(line)

dst.write_text("\n".join(patched) + "\n")
print(f"Patched generated C++ -> {dst}")
PY

    PATCHED_CPP="${dst_cpp}"
}
