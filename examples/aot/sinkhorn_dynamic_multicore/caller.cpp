#ifndef KERNEL_CPP
#define KERNEL_CPP "sinkhorn.cpp"
#endif
#include KERNEL_CPP

extern "C" void call_sinkhorn_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *matrix_in,
    uint8_t *matrix_out,
    uint8_t *mu1_out,
    uint8_t *mu2_out,
    uint32_t N,
    uint32_t K,
    uint32_t L,
    uint32_t order,
    float lr,
    float eps,
    float invK,
    float invL,
    float invK1,
    float invL1)
{
    // Reference fires `blockDim * 2` because each AIC has 2 AIVs and the
    // reference is vector-only. The PTODSL builder targets vector cores too
    // (vector_section), so spawn 2x logical workers per supplied blockDim.
    _kernel<<<blockDim * 2, nullptr, stream>>>(
        reinterpret_cast<half *>(matrix_in),
        reinterpret_cast<half *>(matrix_out),
        reinterpret_cast<half *>(mu1_out),
        reinterpret_cast<half *>(mu2_out),
        static_cast<int32_t>(N),
        static_cast<int32_t>(K),
        static_cast<int32_t>(L),
        static_cast<int32_t>(order),
        lr,
        eps,
        invK,
        invL,
        invK1,
        invL1);
}
