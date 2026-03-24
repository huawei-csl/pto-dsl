#ifndef KERNEL_CPP
#error "KERNEL_CPP must be defined at compile time."
#endif

#include <cstdint>

extern "C" int rtGetC2cCtrlAddr(uint64_t *ctrlAddr, uint32_t *ctrlLen);

#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim,
    void *stream,
    uint8_t *out,
    uint8_t *srcA,
    uint8_t *srcB,
    uint8_t *bias,
    uint8_t *fifoMem)
{
    void *fftsAddr = nullptr;
    uint32_t fftsLen = 0;
    (void)blockDim;
    (void)rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(&fftsAddr), &fftsLen);
    (void)fftsLen;

    LaunchTPushPopMatmulAdd(reinterpret_cast<uint8_t *>(fftsAddr), out, srcA, srcB, bias, fifoMem, stream);
}
