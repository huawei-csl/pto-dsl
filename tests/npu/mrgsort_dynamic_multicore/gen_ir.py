"""Print MLIR IR for the dynamic multicore merge-sort kernel.

Usage: python gen_ir.py <dtype>
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import build_mrgsort_kernel

TILE_LENGTH = 1024
BLOCK_LEN = 64


def fn_name(dtype):
    return f"vec_mrgsort_1d_dynamic_{dtype}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen_ir.py <dtype>", file=sys.stderr)
        sys.exit(1)
    dtype = sys.argv[1]
    module = build_mrgsort_kernel(
        fn_name=fn_name(dtype),
        dtype=dtype,
        tile_length=TILE_LENGTH,
        block_len=BLOCK_LEN,
    )
    print(module)
