#ifndef KERNEL_CPP
#define KERNEL_CPP "inverse_auto_sync.cpp"
#endif
#include KERNEL_CPP

#ifndef KERNEL_FN
#define KERNEL_FN tri_inv_trick_fp16
#endif

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *tensor_out,
    uint8_t *tensor_in,
    uint8_t *identity_in,
    uint32_t matrix_size,
    uint32_t max_block_size)
{
    KERNEL_FN<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<float *>(tensor_out),
        reinterpret_cast<float *>(tensor_in),
        reinterpret_cast<float *>(identity_in),
        static_cast<int32_t>(matrix_size),
        static_cast<int32_t>(max_block_size));
}
