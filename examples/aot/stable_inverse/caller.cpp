#ifndef KERNEL_CPP_16
#define KERNEL_CPP_16 "build_artifacts/rec_unroll_auto_sync_16.cpp"
#endif
#ifndef KERNEL_CPP_32
#define KERNEL_CPP_32 "build_artifacts/rec_unroll_auto_sync_32.cpp"
#endif
#ifndef KERNEL_CPP_64
#define KERNEL_CPP_64 "build_artifacts/rec_unroll_auto_sync_64.cpp"
#endif
#ifndef KERNEL_CPP_128
#define KERNEL_CPP_128 "build_artifacts/rec_unroll_auto_sync_128.cpp"
#endif

#ifndef KERNEL_FN_16
#define KERNEL_FN_16 tri_inv_rec_unroll_fp16_16
#endif
#ifndef KERNEL_FN_32
#define KERNEL_FN_32 tri_inv_rec_unroll_fp16_32
#endif
#ifndef KERNEL_FN_64
#define KERNEL_FN_64 tri_inv_rec_unroll_fp16_64
#endif
#ifndef KERNEL_FN_128
#define KERNEL_FN_128 tri_inv_rec_unroll_fp16_128
#endif

#include KERNEL_CPP_16
#include KERNEL_CPP_32
#include KERNEL_CPP_64
#include KERNEL_CPP_128

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *tensor_out,
    uint8_t *tensor_in,
    uint8_t *minus_identity_in,
    uint32_t matrix_size,
    uint32_t num_matrices,
    uint32_t num_bsnd_heads)
{
    switch (matrix_size) {
    case 16:
        KERNEL_FN_16<<<blockDim, nullptr, stream>>>(
            reinterpret_cast<float *>(tensor_out),
            reinterpret_cast<half *>(tensor_in),
            reinterpret_cast<half *>(minus_identity_in),
            static_cast<int32_t>(matrix_size),
            static_cast<int32_t>(num_matrices),
            static_cast<int32_t>(num_bsnd_heads));
        break;
    case 32:
        KERNEL_FN_32<<<blockDim, nullptr, stream>>>(
            reinterpret_cast<float *>(tensor_out),
            reinterpret_cast<half *>(tensor_in),
            reinterpret_cast<half *>(minus_identity_in),
            static_cast<int32_t>(matrix_size),
            static_cast<int32_t>(num_matrices),
            static_cast<int32_t>(num_bsnd_heads));
        break;
    case 64:
        KERNEL_FN_64<<<blockDim, nullptr, stream>>>(
            reinterpret_cast<float *>(tensor_out),
            reinterpret_cast<half *>(tensor_in),
            reinterpret_cast<half *>(minus_identity_in),
            static_cast<int32_t>(matrix_size),
            static_cast<int32_t>(num_matrices),
            static_cast<int32_t>(num_bsnd_heads));
        break;
    case 128:
        KERNEL_FN_128<<<blockDim, nullptr, stream>>>(
            reinterpret_cast<float *>(tensor_out),
            reinterpret_cast<half *>(tensor_in),
            reinterpret_cast<half *>(minus_identity_in),
            static_cast<int32_t>(matrix_size),
            static_cast<int32_t>(num_matrices),
            static_cast<int32_t>(num_bsnd_heads));
        break;
    default:
        break;
    }
}
