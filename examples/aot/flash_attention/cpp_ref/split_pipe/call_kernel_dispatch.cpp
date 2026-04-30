/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
*/

#include <acl/acl.h>
#include <cstdint>
#include <cstdlib>

#include "fa_performance_kernel.h"
#include "generated_cases.h"
#include "runtime/rt.h"

extern "C" void call_kernel(void *stream, int headSize, int s0, int s1, int tile_s1, bool is_causal, uint8_t *q,
                            uint8_t *k, uint8_t *v, uint8_t *o_out, float *qk_tile_fifo, uint16_t *p_tile_fifo,
                            float *exp_max_ififo, float *pv_tile_fifo, float *global_sum_out, float *exp_max_out,
                            float *o_parts_out)
{
    if (is_causal) {
        return;
    }

    uint64_t ffts_val = 0;
    uint32_t ffts_len = 0;
    rtGetC2cCtrlAddr(&ffts_val, &ffts_len);
    auto *ffts = reinterpret_cast<uint16_t *>(static_cast<uintptr_t>(ffts_val));

    uint8_t *cv_comm_buf = nullptr;

#define LAUNCH_DISPATCH(S0_, HEAD_, S1_, CUBE_S0_, CUBE_S1_, TILE_S1_, QK_PRELOAD_, CAUSAL_MASK_)                         \
    if (headSize == (HEAD_) && (s0) == (S0_) && (s1) == (S1_) && tile_s1 == (TILE_S1_)) {                                 \
        LaunchTFA<(S0_), (HEAD_), (S1_), (CUBE_S0_), (CUBE_S1_), (TILE_S1_), (QK_PRELOAD_), kFaCvFifoSize, false,        \
                  (CAUSAL_MASK_), kFaCvFifoConsSyncPeriod>(                                                             \
            ffts, reinterpret_cast<aclFloat16 *>(q), reinterpret_cast<aclFloat16 *>(k),                                 \
            reinterpret_cast<aclFloat16 *>(v), reinterpret_cast<aclFloat16 *>(p_tile_fifo), exp_max_ififo,             \
            global_sum_out, exp_max_out, reinterpret_cast<float *>(o_out), o_parts_out, qk_tile_fifo, pv_tile_fifo,       \
            reinterpret_cast<aclrtStream>(stream), cv_comm_buf);                                                       \
        return;                                                                                                         \
    }

    TFA_FOR_EACH_CASE(LAUNCH_DISPATCH);

#undef LAUNCH_DISPATCH
}
