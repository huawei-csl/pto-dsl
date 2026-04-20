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
    uint8_t *x,
    uint8_t *y)
{
    void *fftsAddr = nullptr;
    uint32_t fftsLen = 0;
    (void)rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&fftsAddr), &fftsLen);
    (void)fftsLen;

    call_both<<<blockDim, nullptr, stream>>>(
        (__gm__ uint64_t *)fftsAddr,
        (__gm__ float *)gmSlotBuffer,
        (__gm__ float *)x,
        (__gm__ float *)y);
}
