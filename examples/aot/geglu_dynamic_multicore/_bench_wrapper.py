"""
Single-config benchmark wrapper for the agentic optimizer.
Loads geglu_lib.so and prints:  latency_ms=<number>
"""
import ctypes

import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device

# Representative shape — change to target a different operating point
BATCH     = 1024
N_COLS    = 8192
BLOCK_DIM = 24
WARMUP    = 5
ITERS     = 20


def torch_to_ctypes(t):
    return ctypes.c_void_p(t.data_ptr())


device = get_test_device()
torch.npu.set_device(device)

lib = ctypes.CDLL("./geglu_lib.so")
lib.call_kernel.argtypes = [
    ctypes.c_uint32,  # blockDim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # a
    ctypes.c_void_p,  # b
    ctypes.c_void_p,  # c (output)
    ctypes.c_uint32,  # batch
    ctypes.c_uint32,  # n_cols
]
lib.call_kernel.restype = None


def run(a, b, c):
    lib.call_kernel(
        BLOCK_DIM,
        torch.npu.current_stream()._as_parameter_,
        torch_to_ctypes(a),
        torch_to_ctypes(b),
        torch_to_ctypes(c),
        BATCH,
        N_COLS,
    )


dtype = torch.float16
# Separate tensors per iteration to reduce cache reuse
as_ = [torch.randn(BATCH, N_COLS, device=device, dtype=dtype).clamp(-4, 4) for _ in range(WARMUP + ITERS)]
bs_ = [torch.randn(BATCH, N_COLS, device=device, dtype=dtype) for _ in range(WARMUP + ITERS)]
c   = torch.empty(BATCH, N_COLS, device=device, dtype=dtype)

for i in range(WARMUP):
    run(as_[i], bs_[i], c)
torch.npu.synchronize()

starts = [torch.npu.Event(enable_timing=True) for _ in range(ITERS)]
ends   = [torch.npu.Event(enable_timing=True) for _ in range(ITERS)]
for i in range(ITERS):
    starts[i].record()
    run(as_[WARMUP + i], bs_[WARMUP + i], c)
    ends[i].record()
torch.npu.synchronize()

ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / ITERS
print(f"latency_ms={ms:.4f}")
