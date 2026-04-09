"""Print MLIR IR for dynamic multicore col/row expand kernels.

Usage:
  python gen_ir.py --mode colexpand|colexpand_sub|colexpand_div|colexpand_mul|colexpand_min|colexpand_max|rowexpand|rowexpand_mul|rowexpand_sub|rowexpand_div
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from expand_builder import (
    build_col_expand,
    build_col_expand_add,
    build_col_expand_div,
    build_col_expand_expdif,
    build_col_expand_max,
    build_col_expand_min,
    build_col_expand_mul,
    build_col_expand_sub,
    build_row_expand,
    build_row_expand_add,
    build_row_expand_div,
    build_row_expand_expdif,
    build_row_expand_max,
    build_row_expand_min,
    build_row_expand_mul,
    build_row_expand_sub,
)

_BUILDERS = {
    "colexpand": build_col_expand,
    "colexpand_sub": build_col_expand_sub,
    "colexpand_div": build_col_expand_div,
    "colexpand_mul": build_col_expand_mul,
    "colexpand_min": build_col_expand_min,
    "colexpand_max": build_col_expand_max,
    "colexpand_add": build_col_expand_add,
    "colexpand_expdif": build_col_expand_expdif,
    "rowexpand": build_row_expand,
    "rowexpand_add": build_row_expand_add,
    "rowexpand_mul": build_row_expand_mul,
    "rowexpand_sub": build_row_expand_sub,
    "rowexpand_div": build_row_expand_div,
    "rowexpand_min": build_row_expand_min,
    "rowexpand_max": build_row_expand_max,
    "rowexpand_expdif": build_row_expand_expdif,
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
