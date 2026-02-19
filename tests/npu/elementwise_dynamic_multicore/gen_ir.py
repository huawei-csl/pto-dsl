"""Print MLIR IR for a binary op at a given dimension and dtype.

Usage: python gen_ir.py <op_name> <1d|2d> [dtype]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ptodsl.language as pto
from builder import build_binary_kernels

_OPS = {
    "add": pto.add,
    "sub": pto.sub,
    "mul": pto.mul,
    "div": pto.div,
    "or": pto.or_,
}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gen_ir.py <op_name> <1d|2d> [dtype]", file=sys.stderr)
        sys.exit(1)
    op_name = sys.argv[1]
    dim = sys.argv[2]
    dtype = sys.argv[3] if len(sys.argv) > 3 else None

    kernel_1d, kernel_2d = build_binary_kernels(op_name, _OPS[op_name], dtype=dtype)

    if dim == "1d":
        print(kernel_1d)
    elif dim == "2d":
        print(kernel_2d)
    else:
        print(f"Unknown dim: {dim}", file=sys.stderr)
        sys.exit(1)
