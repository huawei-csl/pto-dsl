"""Reference + ctypes wrapper for the deepseek_v4 ``sparse_attn`` PTO kernel.

The NPU kernel implements full FlashAttention with indexed (top-k) KV
gather; this module provides the GPU-semantics reference and the ctypes
shim that the test uses to invoke the compiled kernel.

Per-(b, m) attention with top-k sparse KV:
    q[b, m, h, d]  fp16    H_PAD heads (padded), D = 128
    kv[b, n, d]    fp16    one KV head, N positions per batch
    o[b, m, h, d]  fp16
    attn_sink[H_PAD]      fp32 (per-head additive sink-logit)
    topk_idxs[b, m, K] int32 (indices into the KV n-axis)
"""

import ctypes
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
_KERNEL_SO = _HERE / "sparse_attn_lib.so"

H_PAD = 16
D = 128
BLOCK = 64


def sparse_attn_ref(
    q: torch.Tensor,  # [B, M, H, D] fp16  (H may be < H_PAD)
    kv: torch.Tensor,  # [B, N, D] fp16
    attn_sink: torch.Tensor,  # [H] fp32
    topk_idxs: torch.Tensor,  # [B, M, K] int32 (-1 marks invalid slot)
    scale: float,
) -> torch.Tensor:
    B, M, H, Dq = q.shape
    Bk, N, Dk = kv.shape
    assert (B, Dq) == (Bk, Dk) and H <= H_PAD and Dq == D
    K = topk_idxs.shape[-1]

    qf = q.to(torch.float32)
    kf = kv.to(torch.float32)

    out = torch.zeros_like(qf)
    for b in range(B):
        for m in range(M):
            raw_idx = topk_idxs[b, m]  # [K] int32, may be -1
            invalid = raw_idx == -1  # [K]
            safe_idx = raw_idx.clone().to(torch.long)
            safe_idx[invalid] = 0  # avoid OOB gather
            kv_sel = kf[b, safe_idx]  # [K, D]
            logits = (qf[b, m] @ kv_sel.T) * scale  # [H, K]
            logits[:, invalid] = float("-inf")  # mask sentinel slots
            sink = attn_sink.to(torch.float32).view(H, 1)  # [H, 1]
            logits_full = torch.cat([logits, sink], dim=-1)  # [H, K+1]
            p = torch.softmax(logits_full, dim=-1)
            p_kv = p[:, :K]  # [H, K]
            # Zero contribution from invalid slots (softmax already gave 0
            # because logits were -inf, but be defensive against NaN).
            p_kv = torch.where(invalid.view(1, K), torch.zeros_like(p_kv), p_kv)
            out[b, m] = p_kv @ kv_sel  # [H, D]
    return out.to(torch.float16)


_ARGTYPES = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_float,
]


_lib = None


def _load():
    global _lib
    if _lib is None:
        if not _KERNEL_SO.is_file():
            raise FileNotFoundError(
                f"Kernel shared library not found: {_KERNEL_SO}\n"
                f"Build first: cd {_HERE} && ./compile.sh"
            )
        _lib = ctypes.CDLL(str(_KERNEL_SO))
        _lib.call_kernel.argtypes = _ARGTYPES
        _lib.call_kernel.restype = None
    return _lib


def sparse_attn(q, kv, attn_sink, topk_idxs, scale: float):
    assert q.is_npu and kv.is_npu and q.dtype == kv.dtype == torch.float16
    B, M, H, Dq = q.shape
    Bk, N, Dk = kv.shape
    K = topk_idxs.shape[-1]
    assert H <= H_PAD and Dq == D
    orig_H = H
    # Pad heads to H_PAD: kernel statically expects H == H_PAD == 16.
    if H < H_PAD:
        pad_q = q.new_zeros(B, M, H_PAD - H, Dq)
        q = torch.cat([q, pad_q], dim=2).contiguous()
        pad_sink = attn_sink.new_zeros(H_PAD - H)
        attn_sink = torch.cat([attn_sink, pad_sink]).contiguous()
        H = H_PAD
    o = torch.empty_like(q)
    lib = _load()
    dev = torch.npu.current_device()
    blk = torch.npu.get_device_properties(dev).cube_core_num
    lib.call_kernel(
        blk,
        torch.npu.current_stream()._as_parameter_,
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(kv.data_ptr()),
        ctypes.c_void_p(o.data_ptr()),
        ctypes.c_void_p(attn_sink.data_ptr()),
        ctypes.c_void_p(topk_idxs.data_ptr()),
        ctypes.c_int32(B),
        ctypes.c_int32(M),
        ctypes.c_int32(N),
        ctypes.c_int32(K),
        ctypes.c_float(scale),
    )
    torch.npu.synchronize()
    if orig_H < H_PAD:
        o = o.narrow(2, 0, orig_H).contiguous()
    return o
