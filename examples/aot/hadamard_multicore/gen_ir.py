import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import build_hadamard_kernel


def case_id(dtype, n):
    return f"{dtype}_n{n}"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gen_ir.py <dtype> <n>", file=sys.stderr)
        sys.exit(1)
    dtype = sys.argv[1]
    n = int(sys.argv[2])
    module = build_hadamard_kernel(case_id(dtype, n), dtype, n)
    print(module)
