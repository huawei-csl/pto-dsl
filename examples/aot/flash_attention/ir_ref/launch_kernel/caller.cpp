#ifndef KERNEL_CPP
#error "KERNEL_CPP must be defined at compile time."
#endif

#include <cstdint>

extern "C" int rtGetC2cCtrlAddr(uint64_t *ctrlAddr, uint32_t *ctrlLen);

#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *gmSlotBuffer,
    uint8_t *q,
    uint8_t *k,     // K: [S1_TOTAL, HEAD] fp16
    uint8_t *v,
    uint8_t *o)     // output O: [Q_ROWS, HEAD] fp32
{
    void *fftsAddr = nullptr;
    uint32_t fftsLen = 0;
    (void)rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&fftsAddr), &fftsLen);
    (void)fftsLen;

    call_both<<<blockDim, nullptr, stream>>>(
        (__gm__ int64_t *)fftsAddr,
        (__gm__ float *)gmSlotBuffer,
        (__gm__ half *)q,
        (__gm__ half *)k,
        (__gm__ half *)v,
        (__gm__ float *)o);
}
