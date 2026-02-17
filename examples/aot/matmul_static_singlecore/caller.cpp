#include "matmul.cpp"

extern "C" void call_kernel(
    uint32_t blockDim, void* stream,
    uint8_t* c, uint8_t* a, uint8_t* b)
{
    RunTMATMULSplitK<<<blockDim, nullptr, stream>>>(
      reinterpret_cast<float*>(c),
      reinterpret_cast<float*>(a),
      reinterpret_cast<float*>(b),
      nullptr, false
    );
}
