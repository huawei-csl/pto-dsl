"""Print MLIR IR for the dynamic multicore rsqrt kernel.

Usage: python gen_ir.py [dtype]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsqrt_builder import build_rsqrt

if __name__ == "__main__":
    dtype = sys.argv[1] if len(sys.argv) > 1 else "float32"
    print(build_rsqrt(fn_name=f"rsqrt_{dtype}", dtype=dtype))
