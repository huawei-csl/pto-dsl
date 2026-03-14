#include "mul.cpp"

extern "C" void call_kernel(
    uint32_t blockDim, void* stream,
    uint8_t* c, uint8_t* a, uint8_t* b, uint32_t batch_size)
{
    RunTMATMULSplitK<<<blockDim, nullptr, stream>>>(
      reinterpret_cast<half*>(c),
      reinterpret_cast<half*>(a),
      reinterpret_cast<half*>(b),
      nullptr, false, batch_size
    );
}
