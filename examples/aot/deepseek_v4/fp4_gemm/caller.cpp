#ifndef KERNEL_CPP
#define KERNEL_CPP "fp4_gemm.cpp"
#endif
#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim, void *stream,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *sa, uint8_t *sb,
    int32_t M, int32_t N, int32_t K)
{
    fp4_gemm<<<blockDim, nullptr, stream>>>(
        (__fp16 *)a, (__fp16 *)b, (__fp16 *)c,
        (float *)sa, (float *)sb, M, N, K);
}
