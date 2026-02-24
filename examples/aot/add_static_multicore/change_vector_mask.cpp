#include "pto/pto-inst.hpp"
using namespace pto;

#define NUM_CORES 24  // hard-coded to 910B2 cores

__global__ AICORE void bad_mask() {
    #if defined(__DAV_VEC__)
    set_mask_count();
    set_vector_mask(0, 128);
    #endif // __DAV_VEC__
    return;
}

__global__ AICORE void good_mask() {
    #if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    #endif // __DAV_VEC__
    return;
}

extern "C" void call_bad_mask(
    void *stream)
{
    bad_mask<<<NUM_CORES, nullptr, stream>>>();
}

extern "C" void call_good_mask(
    void *stream)
{
    good_mask<<<NUM_CORES, nullptr, stream>>>();
}
