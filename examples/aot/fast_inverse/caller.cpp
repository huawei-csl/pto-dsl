#ifndef KERNEL_CPP_16
#define KERNEL_CPP_16 "inverse_auto_sync_16.cpp"
#endif
#ifndef KERNEL_CPP_32
#define KERNEL_CPP_32 "inverse_auto_sync_32.cpp"
#endif
#ifndef KERNEL_CPP_64
#define KERNEL_CPP_64 "inverse_auto_sync_64.cpp"
#endif
#ifndef KERNEL_CPP_96
#define KERNEL_CPP_96 "inverse_auto_sync_96.cpp"
#endif
#ifndef KERNEL_CPP_128
#define KERNEL_CPP_128 "inverse_auto_sync_128.cpp"
#endif

#ifndef KERNEL_FN_16
#define KERNEL_FN_16 tri_inv_trick_fp16_16
#endif
#ifndef KERNEL_FN_32
#define KERNEL_FN_32 tri_inv_trick_fp16_32
#endif
#ifndef KERNEL_FN_64
#define KERNEL_FN_64 tri_inv_trick_fp16_64
#endif
#ifndef KERNEL_FN_96
#define KERNEL_FN_96 tri_inv_trick_fp16_96
#endif
#ifndef KERNEL_FN_128
#define KERNEL_FN_128 tri_inv_trick_fp16_128
#endif

#include KERNEL_CPP_16
#include KERNEL_CPP_32
#include KERNEL_CPP_64
#include KERNEL_CPP_96
#include KERNEL_CPP_128

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *tensor_out,
    uint8_t *tensor_in,
    uint8_t *identity_in,
    uint32_t matrix_size,
    uint32_t max_block_size)
{
    switch (matrix_size) {
    case 16:
        KERNEL_FN_16<<<blockDim, nullptr, stream>>>(
            reinterpret_cast<float *>(tensor_out),
            reinterpret_cast<half *>(tensor_in),
            reinterpret_cast<half *>(identity_in),
            static_cast<int32_t>(matrix_size),
            static_cast<int32_t>(max_block_size));
        break;
    case 32:
        KERNEL_FN_32<<<blockDim, nullptr, stream>>>(
            reinterpret_cast<float *>(tensor_out),
            reinterpret_cast<half *>(tensor_in),
            reinterpret_cast<half *>(identity_in),
            static_cast<int32_t>(matrix_size),
            static_cast<int32_t>(max_block_size));
        break;
    case 64:
        KERNEL_FN_64<<<blockDim, nullptr, stream>>>(
            reinterpret_cast<float *>(tensor_out),
            reinterpret_cast<half *>(tensor_in),
            reinterpret_cast<half *>(identity_in),
            static_cast<int32_t>(matrix_size),
            static_cast<int32_t>(max_block_size));
        break;
    case 96:
        KERNEL_FN_96<<<blockDim, nullptr, stream>>>(
            reinterpret_cast<float *>(tensor_out),
            reinterpret_cast<half *>(tensor_in),
            reinterpret_cast<half *>(identity_in),
            static_cast<int32_t>(matrix_size),
            static_cast<int32_t>(max_block_size));
        break;
    case 128:
        KERNEL_FN_128<<<blockDim, nullptr, stream>>>(
            reinterpret_cast<float *>(tensor_out),
            reinterpret_cast<half *>(tensor_in),
            reinterpret_cast<half *>(identity_in),
            static_cast<int32_t>(matrix_size),
            static_cast<int32_t>(max_block_size));
        break;
    default:
        break;
    }
}