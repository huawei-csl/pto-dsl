import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import build_gather_kernel


def _case_id(dtype, rows, cols):
    return f"{dtype}_{rows}x{cols}"


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python gen_ir.py <dtype> <rows> <cols>", file=sys.stderr)
        sys.exit(1)
    dtype = sys.argv[1]
    rows, cols = int(sys.argv[2]), int(sys.argv[3])
    module = build_gather_kernel(_case_id(dtype, rows, cols), dtype, rows, cols)
    print(module)
