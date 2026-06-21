#ifndef KERNEL_CPP
#define KERNEL_CPP "sparse_attn.cpp"
#endif
#include KERNEL_CPP

extern "C" void call_kernel(
    uint32_t blockDim, void *stream,
    uint8_t *q, uint8_t *kv, uint8_t *o,
    uint8_t *attn_sink, uint8_t *topk_idxs,
    int32_t B, int32_t M, int32_t N, int32_t TOPK,
    float scale)
{
    sparse_attn<<<blockDim, nullptr, stream>>>(
        (__fp16 *)q, (__fp16 *)kv, (__fp16 *)o,
        (float *)attn_sink, (int32_t *)topk_idxs,
        B, M, N, TOPK, scale);
}
