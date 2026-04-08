"""Emit MLIR IR for a cvt kernel.

Usage:
  python gen_ir.py --src-dtype float32 --dst-dtype float16 [--rmode round]
  python gen_ir.py --src-dtype float16 --dst-dtype float32
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import _SUPPORTED_DTYPES, build_cvt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dtype", required=True, choices=sorted(_SUPPORTED_DTYPES))
    parser.add_argument("--dst-dtype", required=True, choices=sorted(_SUPPORTED_DTYPES))
    parser.add_argument("--rmode", default=None)
    args = parser.parse_args()

    print(build_cvt(args.src_dtype, args.dst_dtype, rmode=args.rmode))
