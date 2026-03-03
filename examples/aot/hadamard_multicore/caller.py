import sys

_DTYPE_TO_CTYPE = {
    "float16": "__fp16",
    "float32": "float",
}


def generate_caller(dtype):
    src_ctype = _DTYPE_TO_CTYPE[dtype]
    fn = f"{dtype}_dynamic"
    return f"""\
#include "{fn}.cpp"

extern "C" void call_{fn}(
    void *stream, uint8_t *src, uint8_t *out,
    int32_t rows, int32_t cols, int32_t log2_cols)
{{
    {fn}<<<20, nullptr, stream>>>(
        ({src_ctype} *)src, ({src_ctype} *)out, rows, cols, log2_cols);
}}
"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caller.py <dtype>", file=sys.stderr)
        sys.exit(1)
    print(generate_caller(sys.argv[1]))
