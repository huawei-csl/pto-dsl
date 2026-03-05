"""Print MLIR IR for the dynamic multicore rowsum kernel (fp32).

Usage: python gen_ir.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rowsum_builder import build_rowsum

if __name__ == "__main__":
    print(build_rowsum(dtype="fp32"))
