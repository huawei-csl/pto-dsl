"""Emit MLIR IR for a quant kernel.

Usage:
  python gen_ir.py sym
  python gen_ir.py asym
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import build_sym_dynamic, build_asym_dynamic

_BUILDERS = {
    "sym": build_sym_dynamic,
    "asym": build_asym_dynamic,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen_ir.py <sym|asym>", file=sys.stderr)
        sys.exit(1)
    variant = sys.argv[1]
    if variant not in _BUILDERS:
        print(
            f"Unknown variant: {variant!r}. Choose from: {list(_BUILDERS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(_BUILDERS[variant]())
