"""Generate caller.cpp for the dynamic multicore gather kernel."""

import sys

_DTYPE_TO_CTYPE = {
    "float32": "float",
    "float16": "half",
    "int16": "int16_t",
    "int32": "int32_t",
}


def fn_name(dtype, mask_pattern="P1111"):
    return f"vec_gather_2d_dynamic_{dtype}_{mask_pattern}"


def generate_caller(dtype, mask_pattern="P1111"):
    src_ctype = _DTYPE_TO_CTYPE[dtype]
    fn = fn_name(dtype, mask_pattern)
    return f"""\
#include "{fn}.cpp"

extern "C" void call_{fn}(
    void *stream, uint8_t *src, uint8_t *indices, uint8_t *out, int32_t N)
{{
    {fn}<<<20, nullptr, stream>>>(
        ({src_ctype} *)src, (int32_t *)indices, ({src_ctype} *)out, N);
}}
"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caller.py <dtype> [mask_pattern]", file=sys.stderr)
        sys.exit(1)
    dtype = sys.argv[1]
    mask_pattern = sys.argv[2] if len(sys.argv) > 2 else "P1111"
    print(generate_caller(dtype, mask_pattern))
