#ifndef KERNEL_CPP
#define KERNEL_CPP "geglu.cpp"
#endif
#include KERNEL_CPP

#ifndef NUM_CORES
#define NUM_CORES 24
#endif

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *a,
    uint8_t *b,
    uint8_t *c,
    uint32_t batch,
    uint32_t n_cols)
{
    uint32_t launch_blocks = blockDim > 0 ? blockDim : NUM_CORES;
    _kernel<<<launch_blocks, nullptr, stream>>>(
        reinterpret_cast<half *>(a),
        reinterpret_cast<half *>(b),
        reinterpret_cast<half *>(c),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n_cols));
}
