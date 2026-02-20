import sys

_DTYPE_TO_CTYPE = {
    "float32": "float",
    "float16": "half",
}


def case_id(dtype, rows, cols):
    return f"{dtype}_{rows}x{cols}"


def generate_caller(dtype, rows, cols):
    src_ctype = _DTYPE_TO_CTYPE[dtype]
    fn = case_id(dtype, rows, cols)
    return f"""\
#include "{fn}.cpp"

extern "C" void call_{fn}(
    void *stream, uint8_t *src, uint8_t *indices, uint8_t *out)
{{
    {fn}<<<20, nullptr, stream>>>(
        ({src_ctype} *)src, (int32_t *)indices, ({src_ctype} *)out);
}}
"""


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python caller.py <dtype> <rows> <cols>", file=sys.stderr)
        sys.exit(1)
    print(generate_caller(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])))
