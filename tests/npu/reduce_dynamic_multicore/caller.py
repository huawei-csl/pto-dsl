"""Generate caller.cpp for dynamic multicore row/col reduction kernels.

Usage:
  python caller.py --mode rowsum|rowmin|rowmax|rowprod|colsum|colmin|colmax|colprod
"""


def generate_caller(mode, dtype):
    ctype = "half" if dtype == "fp16" else "float"
    return f"""\
#include "{mode}.cpp"

extern "C" void call_{mode}(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint32_t batch,
    uint32_t n_cols)
{{
    _kernel<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<{ctype} *>(x),
        reinterpret_cast<{ctype} *>(y),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n_cols));
}}
"""


if __name__ == "__main__":
    import argparse

    MODES = [
        "rowsum",
        "rowmin",
        "rowmax",
        "rowprod",
        "colsum",
        "colmin",
        "colmax",
        "colprod",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()

    print(generate_caller(args.mode, args.dtype))
