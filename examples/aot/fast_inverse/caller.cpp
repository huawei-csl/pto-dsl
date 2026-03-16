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
#ifdef MANUAL_SYNC
#include "fast_inverse_s16_manual_sync.cpp"
#include "fast_inverse_s32_manual_sync.cpp"
#include "fast_inverse_s64_manual_sync.cpp"
#include "fast_inverse_s96_manual_sync.cpp"
#include "fast_inverse_s128_manual_sync.cpp"
#else
#include "fast_inverse_s16_auto_sync.cpp"
#include "fast_inverse_s32_auto_sync.cpp"
#include "fast_inverse_s64_auto_sync.cpp"
#include "fast_inverse_s96_auto_sync.cpp"
#include "fast_inverse_s128_auto_sync.cpp"
#endif

#include <cstdint>
#include <stdexcept>

extern "C" void call_kernel(
    uint32_t blockDim,
    void* stream,
    uint8_t* out_ptr,
    uint8_t* in_ptr,
    uint8_t* i_neg_ptr,
    uint32_t matrix_size,
    uint32_t max_block_size)
{
    const uint32_t launch_blocks = blockDim > 0 ? blockDim : 1;

    switch (matrix_size) {
    case 16:
#ifdef MANUAL_SYNC
        fast_inverse_s16_manualsync<<<launch_blocks, nullptr, stream>>>(
#else
        fast_inverse_s16_autosync<<<launch_blocks, nullptr, stream>>>(
#endif
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<half*>(in_ptr),
            reinterpret_cast<half*>(i_neg_ptr),
            static_cast<int32_t>(max_block_size));
        break;
    case 32:
#ifdef MANUAL_SYNC
        fast_inverse_s32_manualsync<<<launch_blocks, nullptr, stream>>>(
#else
        fast_inverse_s32_autosync<<<launch_blocks, nullptr, stream>>>(
#endif
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<half*>(in_ptr),
            reinterpret_cast<half*>(i_neg_ptr),
            static_cast<int32_t>(max_block_size));
        break;
    case 64:
#ifdef MANUAL_SYNC
        fast_inverse_s64_manualsync<<<launch_blocks, nullptr, stream>>>(
#else
        fast_inverse_s64_autosync<<<launch_blocks, nullptr, stream>>>(
#endif
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<half*>(in_ptr),
            reinterpret_cast<half*>(i_neg_ptr),
            static_cast<int32_t>(max_block_size));
        break;
    case 96:
#ifdef MANUAL_SYNC
        fast_inverse_s96_manualsync<<<launch_blocks, nullptr, stream>>>(
#else
        fast_inverse_s96_autosync<<<launch_blocks, nullptr, stream>>>(
#endif
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<half*>(in_ptr),
            reinterpret_cast<half*>(i_neg_ptr),
            static_cast<int32_t>(max_block_size));
        break;
    case 128:
#ifdef MANUAL_SYNC
        fast_inverse_s128_manualsync<<<launch_blocks, nullptr, stream>>>(
#else
        fast_inverse_s128_autosync<<<launch_blocks, nullptr, stream>>>(
#endif
            reinterpret_cast<float*>(out_ptr),
            reinterpret_cast<half*>(in_ptr),
            reinterpret_cast<half*>(i_neg_ptr),
            static_cast<int32_t>(max_block_size));
        break;
    default:
        throw std::runtime_error("Unsupported matrix_size for fast_inverse kernel");
    }
}
