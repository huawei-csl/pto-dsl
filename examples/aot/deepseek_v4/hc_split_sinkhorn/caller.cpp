#ifndef KERNEL_CPP
#define KERNEL_CPP "hc_split_sinkhorn.cpp"
#endif
#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim, void *stream,
    uint8_t *mixes,
    uint8_t *hc_scale,
    uint8_t *hc_base,
    uint8_t *pre,
    uint8_t *post,
    uint8_t *comb,
    int32_t n)
{
    hc_split_sinkhorn<<<blockDim, nullptr, stream>>>(
        (float *)mixes,
        (float *)hc_scale,
        (float *)hc_base,
        (float *)pre,
        (float *)post,
        (float *)comb,
        n);
}
