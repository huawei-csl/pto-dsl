"""Generate caller.cpp for the dynamic multicore rsqrt kernel.

Usage: python caller.py [dtype]
"""

import sys

_DTYPE_TO_CTYPE = {
    "float32": "float",
    "float16": "half",
}

_BLOCK_DIM = 24


def generate_caller(dtype="float32"):
    ctype = _DTYPE_TO_CTYPE[dtype]
    return f"""\
#include "rsqrt_{dtype}.cpp"

extern "C" void call_kernel(
    void *stream, uint8_t *x, uint8_t *y, int32_t batch, int32_t n_cols)
{{
    _kernel<<<{_BLOCK_DIM}, nullptr, stream>>>(
        ({ctype} *)x, ({ctype} *)y, batch, n_cols);
}}
"""


if __name__ == "__main__":
    dtype = sys.argv[1] if len(sys.argv) > 1 else "float32"
    print(generate_caller(dtype=dtype))
