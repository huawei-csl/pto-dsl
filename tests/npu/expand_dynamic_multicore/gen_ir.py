"""Print MLIR IR for dynamic multicore col/row expand kernels.

Usage:
  python gen_ir.py --mode colexpand|rowexpand|rowexpand_mul|rowexpand_sub|rowexpand_div
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expand_builder import (
    build_col_expand,
    build_row_expand,
    build_row_expand_div,
    build_row_expand_mul,
    build_row_expand_sub,
)

_BUILDERS = {
    "colexpand": build_col_expand,
    "rowexpand": build_row_expand,
    "rowexpand_mul": build_row_expand_mul,
    "rowexpand_sub": build_row_expand_sub,
    "rowexpand_div": build_row_expand_div,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(_BUILDERS.keys()),
        default="colexpand",
    )
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()

    print(_BUILDERS[args.mode](dtype=args.dtype))
