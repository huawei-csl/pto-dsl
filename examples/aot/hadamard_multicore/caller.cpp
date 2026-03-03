#include "float16_dynamic_edited_full.cpp"

extern "C" void call_float16_dynamic(
    void *stream, uint8_t *src, uint8_t *out,
    int32_t rows, int32_t cols, int32_t log2_cols)
{
    float16_dynamic<<<20, nullptr, stream>>>(
        (__fp16 *)src, (__fp16 *)out, rows, cols, log2_cols);
}

