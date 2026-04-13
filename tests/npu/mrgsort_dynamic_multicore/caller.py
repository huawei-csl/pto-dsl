"""Generate caller.cpp for the dynamic multicore merge-sort kernel."""

import sys

_DTYPE_TO_CTYPE = {
    "float32": "float",
    "float16": "half",
}


def fn_name(dtype):
    return f"vec_mrgsort_1d_dynamic_{dtype}"


def generate_caller(dtype):
    ctype = _DTYPE_TO_CTYPE[dtype]
    fn = fn_name(dtype)
    return f"""\
#include "{fn}.cpp"

extern "C" void call_{fn}(
    uint32_t blockDim, void *stream, uint8_t *src, uint8_t *out, int32_t N)
{{
    {fn}<<<blockDim, nullptr, stream>>>(
        ({ctype} *)src, ({ctype} *)out, (int32_t)N);
}}
"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caller.py <dtype>", file=sys.stderr)
        sys.exit(1)
    dtype = sys.argv[1]
    print(generate_caller(dtype))
