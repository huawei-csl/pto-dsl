#include "float16_n32.cpp"

extern "C" void call_float16_n32(
    void *stream, uint8_t *src, uint8_t *out, int32_t rows)
{
    float16_n32<<<20, nullptr, stream>>>(
        (__fp16 *)src, (__fp16 *)out, rows);
}

