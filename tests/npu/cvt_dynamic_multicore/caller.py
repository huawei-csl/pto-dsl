"""Generate caller.cpp for cvt kernels."""

import argparse

_BLOCK_DIM = 20

_CPP_TYPE = {
    "float32": "float",
    "float16": "half",
    "int64": "int64_t",
    "int32": "int32_t",
    "int16": "int16_t",
    "int8": "int8_t",
    "uint8": "uint8_t",
}


def generate_caller(src_dtype, dst_dtype, rmode=None):
    stem = f"cvt_{src_dtype}_to_{dst_dtype}" + (f"_{rmode}" if rmode else "")
    fn_name = f"call_cvt_{src_dtype}_to_{dst_dtype}"
    src_cpp = _CPP_TYPE[src_dtype]
    dst_cpp = _CPP_TYPE[dst_dtype]
    return f"""\
#include "{stem}.cpp"

extern "C" void {fn_name}(
    void *stream,
    uint8_t *src,
    uint8_t *dst,
    int32_t batch,
    int32_t n_cols)
{{
    _kernel<<<{_BLOCK_DIM}, nullptr, stream>>>(
        reinterpret_cast<{src_cpp} *>(src),
        reinterpret_cast<{dst_cpp} *>(dst),
        batch,
        n_cols);
}}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dtype", required=True)
    parser.add_argument("--dst-dtype", required=True)
    parser.add_argument("--rmode", default=None)
    args = parser.parse_args()

    print(generate_caller(args.src_dtype, args.dst_dtype, rmode=args.rmode or None))
