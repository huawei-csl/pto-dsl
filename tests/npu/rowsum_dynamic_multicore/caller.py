"""Generate caller.cpp for the dynamic multicore rowsum kernel (fp32).

Usage: python caller.py
"""

_BLOCK_DIM = 24


def generate_caller():
    return f"""\
#include "rowsum.cpp"

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint32_t batch,
    uint32_t n_cols)
{{
    _kernel<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<float *>(x),
        reinterpret_cast<float *>(y),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n_cols));
}}
"""


if __name__ == "__main__":
    print(generate_caller())
