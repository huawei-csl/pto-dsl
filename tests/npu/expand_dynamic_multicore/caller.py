"""Generate caller.cpp for dynamic multicore col/row expand kernels.

Usage:
  python caller.py --mode colexpand|rowexpand|rowexpand_mul|rowexpand_sub|rowexpand_div
"""

_FUSED_MODES = {
    "colexpand_add",
    "colexpand_sub",
    "colexpand_div",
    "colexpand_mul",
    "colexpand_min",
    "colexpand_max",
    "colexpand_expdif",
    "rowexpand_add",
    "rowexpand_mul",
    "rowexpand_sub",
    "rowexpand_div",
    "rowexpand_min",
    "rowexpand_max",
    "rowexpand_expdif",
}


def generate_caller(mode, dtype):
    ctype = "half" if dtype == "fp16" else "float"
    if mode in _FUSED_MODES:
        return f"""\
#include "{mode}.cpp"

extern "C" void call_{mode}(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *z,
    uint32_t batch,
    uint32_t n_cols)
{{
    _kernel<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<{ctype} *>(x),
        reinterpret_cast<{ctype} *>(y),
        reinterpret_cast<{ctype} *>(z),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n_cols));
}}
"""
    return f"""\
#include "{mode}.cpp"

extern "C" void call_{mode}(
    uint32_t blockDim,
    void *stream,
    uint8_t *src,
    uint8_t *dst,
    uint32_t batch,
    uint32_t n_cols)
{{
    _kernel<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<{ctype} *>(src),
        reinterpret_cast<{ctype} *>(dst),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n_cols));
}}
"""


if __name__ == "__main__":
    import argparse

    MODES = [
        "colexpand",
        "colexpand_sub",
        "colexpand_div",
        "colexpand_mul",
        "colexpand_min",
        "colexpand_max",
        "colexpand_add",
        "colexpand_expdif",
        "rowexpand",
        "rowexpand_add",
        "rowexpand_mul",
        "rowexpand_sub",
        "rowexpand_div",
        "rowexpand_min",
        "rowexpand_max",
        "rowexpand_expdif",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
    args = parser.parse_args()

    print(generate_caller(args.mode, args.dtype))
