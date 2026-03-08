#include "matmul.cpp"

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *z,
    int M,
    int N,
    int K)
{
    matmul_kernel_ABt<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<half *>(x),
        reinterpret_cast<half *>(y),
        reinterpret_cast<half *>(z),
        static_cast<int32_t>(M),
        static_cast<int32_t>(N),
        static_cast<int32_t>(K));
}
