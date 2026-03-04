#ifndef KERNEL_CPP
#define KERNEL_CPP "hadamard_auto_sync.cpp"
#endif
#include KERNEL_CPP

#ifndef KERNEL_FN
#define KERNEL_FN fast_hadamard_autosync
#endif

#ifndef NUM_CORES
#define NUM_CORES 24
#endif

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint32_t batch,
    uint32_t n,
    uint32_t log2_n)
{
    uint32_t launch_blocks = blockDim > 0 ? blockDim : NUM_CORES;
    KERNEL_FN<<<launch_blocks, nullptr, stream>>>(
        reinterpret_cast<half *>(x),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n),
        static_cast<int32_t>(log2_n));
}
