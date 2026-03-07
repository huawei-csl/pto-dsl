"""Print MLIR IR for a unary op at a given dtype.

Usage: python gen_ir.py <op_name> [dtype]
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ptodsl import tile
from unary_builder import build_unary_kernel

_OPS = {
    "rsqrt": tile.rsqrt,
    "sqrt": tile.sqrt,
    "exp": tile.exp,
    "log": tile.log,
    "relu": tile.relu,
    "abs": tile.abs,
    "reciprocal": tile.reciprocal,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen_ir.py <op_name> [dtype]", file=sys.stderr)
        sys.exit(1)
    op_name = sys.argv[1]
    dtype = sys.argv[2] if len(sys.argv) > 2 else "float32"

    if op_name not in _OPS:
        print(f"Unknown op: {op_name}. Available: {list(_OPS)}", file=sys.stderr)
        sys.exit(1)

    print(build_unary_kernel(op_name, _OPS[op_name], dtype=dtype))
