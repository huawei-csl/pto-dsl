"""Print MLIR IR for the dynamic multicore gather kernel.

Usage: python gen_ir.py <dtype> [mask_pattern]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import build_gather_kernel

def fn_name(dtype, mask_pattern="P1111"):
    return f"vec_gather_2d_dynamic_{dtype}_{mask_pattern}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen_ir.py <dtype> [mask_pattern]", file=sys.stderr)
        sys.exit(1)
    dtype = sys.argv[1]
    mask_pattern = sys.argv[2] if len(sys.argv) > 2 else "P1111"
    module = build_gather_kernel(fn_name=fn_name(dtype, mask_pattern), dtype=dtype, mask_pattern=mask_pattern)
    print(module)
