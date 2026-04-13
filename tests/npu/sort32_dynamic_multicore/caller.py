"""Generate caller.cpp for the dynamic multicore TSORT32 kernel."""

import sys

_DTYPE_TO_CTYPE = {
    "float16": "half",
    "float32": "float",
}


def fn_name(dtype):
    return f"tsort32_1d_dynamic_{dtype}"


def generate_caller(dtype):
    ctype = _DTYPE_TO_CTYPE[dtype]
    fn = fn_name(dtype)
    return f"""\
#include "{fn}.cpp"

extern "C" void call_{fn}(
    uint32_t blockDim, void *stream, uint8_t *src, uint8_t *idx, uint8_t *dst, int32_t N)
{{
    {fn}<<<blockDim, nullptr, stream>>>(
        ({ctype} *)src, (uint32_t *)idx, ({ctype} *)dst, (int32_t)N);
}}
"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caller.py <dtype>", file=sys.stderr)
        sys.exit(1)
    print(generate_caller(sys.argv[1]))
