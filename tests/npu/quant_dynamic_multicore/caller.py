"""Generate caller.cpp for quant dynamic multicore kernels."""

import sys

_BLOCK_DIM = 20


def generate_caller(variant):
    stem = f"quant_{variant}_dynamic"
    fn_name = f"call_{stem}"
    if variant == "sym":
        return f"""\
#include "{stem}.cpp"

extern "C" void {fn_name}(
    void *stream,
    uint8_t *src,
    uint8_t *fp,
    uint8_t *dst,
    int32_t batch,
    int32_t n_cols)
{{
    quant_sym_dynamic<<<{_BLOCK_DIM}, nullptr, stream>>>(
        reinterpret_cast<float *>(src),
        reinterpret_cast<float *>(fp),
        reinterpret_cast<int8_t *>(dst),
        batch,
        n_cols);
}}
"""
    elif variant == "asym":
        return f"""\
#include "{stem}.cpp"

extern "C" void {fn_name}(
    void *stream,
    uint8_t *src,
    uint8_t *fp,
    uint8_t *offset,
    uint8_t *dst,
    int32_t batch,
    int32_t n_cols)
{{
    quant_asym_dynamic<<<{_BLOCK_DIM}, nullptr, stream>>>(
        reinterpret_cast<float *>(src),
        reinterpret_cast<float *>(fp),
        reinterpret_cast<float *>(offset),
        reinterpret_cast<uint8_t *>(dst),
        batch,
        n_cols);
}}
"""
    else:
        raise ValueError(f"Unknown variant: {variant!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python caller.py <sym|asym>", file=sys.stderr)
        sys.exit(1)
    print(generate_caller(sys.argv[1]))
