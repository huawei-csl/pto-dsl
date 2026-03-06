#ifndef KERNEL_CPP
#define KERNEL_CPP "rms_norm.cpp"
#endif
#include KERNEL_CPP

#ifndef NUM_CORES
#define NUM_CORES 24
#endif

#ifndef DTYPE_T
#define DTYPE_T half
#endif

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *w,
    uint8_t *y,
    uint32_t batch,
    uint32_t n_cols)
{
    uint32_t launch_blocks = blockDim > 0 ? blockDim : NUM_CORES;
    _kernel<<<launch_blocks, nullptr, stream>>>(
        reinterpret_cast<DTYPE_T *>(x),
        reinterpret_cast<DTYPE_T *>(w),
        reinterpret_cast<DTYPE_T *>(y),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n_cols));
}
