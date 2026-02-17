#include "div.cpp"

extern "C" void call_kernel(
    void *stream, uint8_t *x, uint8_t *y, uint8_t *z, int32_t N)
{
    vec_div_1d_dynamic<<<20, nullptr, stream>>>(
        (float *)x, (float *)y, (float *)z, N
        );
}
