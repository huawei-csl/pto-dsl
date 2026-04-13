#ifndef KERNEL_CPP
#define KERNEL_CPP "hadamard_auto_sync.cpp"
#endif
#include KERNEL_CPP

#ifndef KERNEL_FN
#define KERNEL_FN fast_hadamard_autosync
#endif

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint32_t batch,
    uint32_t n,
    uint32_t log2_n)
{
    KERNEL_FN<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<half *>(x),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n),
        static_cast<int32_t>(log2_n));
}
