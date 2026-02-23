import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import build_gather_kernel


def _case_id(dtype, rows, cols, mask_pattern="P1111"):
    return f"{dtype}_{rows}x{cols}_{mask_pattern}"


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python gen_ir.py <dtype> <rows> <cols> [mask_pattern]",
            file=sys.stderr,
        )
        sys.exit(1)
    dtype = sys.argv[1]
    rows, cols = int(sys.argv[2]), int(sys.argv[3])
    mask_pattern = sys.argv[4] if len(sys.argv) > 4 else "P1111"
    module = build_gather_kernel(
        _case_id(dtype, rows, cols, mask_pattern), dtype, rows, cols, mask_pattern
    )
    print(module)
