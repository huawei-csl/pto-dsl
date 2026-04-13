#ifndef KERNEL_CPP
#define KERNEL_CPP "add.cpp"
#endif
#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, int32_t N)
{
    vec_add_1d_dynamic<<<blockDim, nullptr, stream>>>(
        (float *)x, (float *)y, (float *)z, N
        );
}
