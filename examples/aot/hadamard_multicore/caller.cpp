    #include "float32_n32.cpp"

    extern "C" void call_float32_n32(
        void *stream, uint8_t *src, uint8_t *out)
    {
        float32_n32<<<20, nullptr, stream>>>(
            (float *)src, (float *)out);
    }
    
