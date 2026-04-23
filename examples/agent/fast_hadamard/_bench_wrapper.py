"""
Single-config benchmark wrapper for the agentic optimizer.
Loads hadamard_auto_sync_lib.so and prints:  latency_ms=<number>
"""
import ctypes
import math

import torch
import torch_npu  # noqa: F401

from ptodsl.test_util import get_test_device

# Representative shape — change to target a different operating point
BATCH     = 32
N         = 8192
BLOCK_DIM = 24
WARMUP    = 5
ITERS     = 20

LOG2_N = int(math.log2(N))


def torch_to_ctypes(t):
    return ctypes.c_void_p(t.data_ptr())


device = get_test_device()
torch.npu.set_device(device)

lib = ctypes.CDLL("./hadamard_auto_sync_lib.so")
lib.call_kernel.argtypes = [
    ctypes.c_uint32,  # blockDim
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # x (in-place)
    ctypes.c_uint32,  # batch
    ctypes.c_uint32,  # n
    ctypes.c_uint32,  # log2_n
]
lib.call_kernel.restype = None


def run(x):
    lib.call_kernel(
        BLOCK_DIM,
        torch.npu.current_stream()._as_parameter_,
        torch_to_ctypes(x),
        BATCH,
        N,
        LOG2_N,
    )


# Allocate separate tensors to avoid cache reuse
xs = [torch.randn(BATCH, N, device=device, dtype=torch.float16) for _ in range(WARMUP + ITERS)]

for i in range(WARMUP):
    run(xs[i])
torch.npu.synchronize()

starts = [torch.npu.Event(enable_timing=True) for _ in range(ITERS)]
ends   = [torch.npu.Event(enable_timing=True) for _ in range(ITERS)]
for i in range(ITERS):
    starts[i].record()
    run(xs[WARMUP + i])
    ends[i].record()
torch.npu.synchronize()

ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / ITERS
print(f"latency_ms={ms:.4f}")
