#ifndef KERNEL_CPP
#define KERNEL_CPP "inverse.cpp"
#endif

#ifndef KERNEL_FN
#define KERNEL_FN tri_inv_block2x2_fp16
#endif

#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *tensor_out,
    uint8_t *tensor_in,
    uint8_t *identity_in,
    uint32_t matrix_size,
    uint32_t log2_blocksize)
{
    (void)matrix_size;
    KERNEL_FN<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<float *>(tensor_out),
        reinterpret_cast<half *>(tensor_in),
        reinterpret_cast<half *>(identity_in),
        static_cast<int32_t>(matrix_size),
        static_cast<int32_t>(log2_blocksize));
}
