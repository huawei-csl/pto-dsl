"""Generate caller.cpp for a given unary op name."""

import sys

_DTYPE_TO_CTYPE = {
    "float32": "float",
    "float16": "half",
    "int32": "int32_t",
    "int16": "int16_t",
}

_BLOCK_DIM = 24


def generate_caller(op_name, dtype="float32"):
    ctype = _DTYPE_TO_CTYPE[dtype]
    return f"""\
#include "{op_name}_{dtype}.cpp"

extern "C" void call_kernel(
    void *stream, uint8_t *x, uint8_t *y, int32_t batch, int32_t n_cols)
{{
    _kernel<<<{_BLOCK_DIM}, nullptr, stream>>>(
        ({ctype} *)x, ({ctype} *)y, batch, n_cols);
}}
"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caller.py <op_name> [dtype]", file=sys.stderr)
        sys.exit(1)
    op_name = sys.argv[1]
    dtype = sys.argv[2] if len(sys.argv) > 2 else "float32"
    print(generate_caller(op_name, dtype))
