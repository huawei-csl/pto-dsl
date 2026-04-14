#ifndef KERNEL_CPP
#define KERNEL_CPP "hadamard_quant.cpp"
#endif
#include KERNEL_CPP

#ifndef KERNEL_FN
#define KERNEL_FN fast_hadamard_quant_autosync
#endif

// The DSL-generated kernel takes explicit has_group_scales / has_group_offsets
// flags (int32).  Derive them here from whether the caller passed a non-null
// pointer so the Python / C++ API stays pointer-based (nullptr == absent).
extern "C" void call_fused_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint8_t *group_scales,
    uint8_t *group_offsets,
    uint32_t scale_group_stride,
    uint32_t offset_group_stride,
    uint32_t batch,
    uint32_t n,
    uint32_t log2_n,
    float scale,
    uint32_t group_size,
    float q_offset)
{
    KERNEL_FN<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<half *>(x),
        reinterpret_cast<int8_t *>(y),
        reinterpret_cast<half *>(group_scales),
        reinterpret_cast<half *>(group_offsets),
        static_cast<int32_t>(scale_group_stride),
        static_cast<int32_t>(offset_group_stride),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n),
        static_cast<int32_t>(log2_n),
        scale,
        static_cast<int32_t>(group_size),
        q_offset,
        group_scales  != nullptr ? 1 : 0,   // has_group_scales
        group_offsets != nullptr ? 1 : 0);  // has_group_offsets
}
