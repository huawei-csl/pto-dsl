#ifndef KERNEL_CPP
#define KERNEL_CPP "geglu.cpp"
#endif
#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *a,
    uint8_t *b,
    uint8_t *c,
    uint32_t batch,
    uint32_t n_cols)
{
    _kernel<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<half *>(a),
        reinterpret_cast<half *>(b),
        reinterpret_cast<half *>(c),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n_cols));
}
