"""Print MLIR IR for dynamic multicore row/col reduction kernels.

Usage:
  python gen_ir.py --mode rowsum|rowmin|rowmax|rowprod|colsum|colmin|colmax|colprod
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reduce_builder import (
    build_colmax,
    build_colmin,
    # build_colprod,
    build_colsum,
    build_rowmax,
    build_rowmin,
    # build_rowprod,
    build_rowsum,
)

_BUILDERS = {
    "rowsum": build_rowsum,
    "rowmin": build_rowmin,
    "rowmax": build_rowmax,
    # "rowprod": build_rowprod,
    "colsum": build_colsum,
    "colmin": build_colmin,
    "colmax": build_colmax,
    # "colprod": build_colprod,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=list(_BUILDERS.keys()),
        default="rowsum",
    )
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()

    print(_BUILDERS[args.mode](dtype=args.dtype))
