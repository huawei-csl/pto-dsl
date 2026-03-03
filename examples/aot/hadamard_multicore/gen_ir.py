import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import build_hadamard_kernel_dynamic


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen_ir.py <dtype>", file=sys.stderr)
        sys.exit(1)
    dtype = sys.argv[1]
    print(build_hadamard_kernel_dynamic(f"{dtype}_dynamic", dtype))
