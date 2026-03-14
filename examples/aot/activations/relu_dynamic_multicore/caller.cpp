#include "generated_relu.cpp"

extern "C" void call_kernel( uint32_t blockDim, void* stream, void* v1, void* v2, uint32_t n) {
    sync_kernel_dyn<<<blockDim, nullptr, stream>>>(( __gm__ float *)v1, (__gm__ float *)v2, n);
}
