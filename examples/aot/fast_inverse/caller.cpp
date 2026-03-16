#ifndef KERNEL_CPP
#define KERNEL_CPP "tri_inv_trick_auto_sync.cpp"
#endif
#include KERNEL_CPP

#ifndef KERNEL_FN
#define KERNEL_FN tri_inv_trick_fp16_autosync
#endif

#ifndef NUM_CORES
#define NUM_CORES 24
#endif

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *out,
    uint8_t *m,
    uint8_t *i_neg,
    uint32_t matrix_size,
    uint32_t max_block_size)
{
    uint32_t launch_blocks = blockDim > 0 ? blockDim : NUM_CORES;
    KERNEL_FN<<<launch_blocks, nullptr, stream>>>(
        reinterpret_cast<float *>(out),
        reinterpret_cast<half *>(m),
        reinterpret_cast<half *>(i_neg),
        static_cast<int32_t>(matrix_size),
        static_cast<int32_t>(max_block_size));
}
