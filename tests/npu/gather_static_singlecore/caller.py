import sys

_DTYPE_TO_CTYPE = {
    "float32": "float",
    "float16": "half",
    "int16": "int16_t",
    "int32": "int32_t",
}


def case_id(dtype, rows, cols, mask_pattern="P1111"):
    return f"{dtype}_{rows}x{cols}_{mask_pattern}"


def generate_caller(dtype, rows, cols, mask_pattern="P1111"):
    src_ctype = _DTYPE_TO_CTYPE[dtype]
    fn = case_id(dtype, rows, cols, mask_pattern)
    return f"""\
#include "{fn}.cpp"

extern "C" void call_{fn}(
    void *stream, uint8_t *src, uint8_t *indices, uint8_t *out)
{{
    {fn}<<<1, nullptr, stream>>>(
        ({src_ctype} *)src, (int32_t *)indices, ({src_ctype} *)out);
}}
"""


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python caller.py <dtype> <rows> <cols> [mask_pattern]",
            file=sys.stderr,
        )
        sys.exit(1)
    mask_pattern = sys.argv[4] if len(sys.argv) > 4 else "P1111"
    print(
        generate_caller(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), mask_pattern)
    )
