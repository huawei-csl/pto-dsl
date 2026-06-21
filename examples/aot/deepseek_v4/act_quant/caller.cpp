#ifndef KERNEL_CPP
#define KERNEL_CPP "act_quant.cpp"
#endif
#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *y, uint8_t *scale,
    int32_t M, int32_t N)
{
    act_quant<<<blockDim, nullptr, stream>>>(
        (__fp16 *)x, (int8_t *)y, (float *)scale, M, N);
}
