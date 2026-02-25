#ifndef KERNEL_CPP
#define KERNEL_CPP "add.cpp"
#endif
#include KERNEL_CPP

#define NUM_CORES 24  // hard-coded to 910B2

extern "C" void call_kernel(
    void *stream, uint8_t *x, uint8_t *y, uint8_t *z, int32_t N)
{
    vec_add_1d_dynamic<<<NUM_CORES, nullptr, stream>>>(
        (float *)x, (float *)y, (float *)z, N
        );
}
