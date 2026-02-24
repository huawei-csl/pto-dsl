#include "vec_gather_2d_dynamic_int32_P1111.cpp"

extern "C" void call_vec_gather_2d_dynamic_int32_P1111(
    void *stream, uint8_t *src, uint8_t *indices, uint8_t *out, int32_t N)
{
    vec_gather_2d_dynamic_int32_P1111<<<20, nullptr, stream>>>(
        (int32_t *)src, (int32_t *)indices, (int32_t *)out, N);
}

