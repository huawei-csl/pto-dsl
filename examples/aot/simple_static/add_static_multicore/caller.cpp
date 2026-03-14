#include "add.cpp"

extern "C" void call_kernel(
    void *stream, uint8_t *x, uint8_t *y, uint8_t *z, int32_t vrow, int32_t vcol)
{
    vec_add_kernel_2d_dynamic<<<20, nullptr, stream>>>(
        (float *)x, (float *)y, (float *)z, vrow, vcol
        );
}
