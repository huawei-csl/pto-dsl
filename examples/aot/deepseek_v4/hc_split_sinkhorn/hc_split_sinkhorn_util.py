"""Reference + ctypes wrapper for the deepseek_v4 ``hc_split_sinkhorn`` kernel.

The full GPU op is now executed entirely on-device: pre/post sigmoid heads
and the comb sinkhorn iteration are all inside one PTO vector_section.
"""

import ctypes
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
_KERNEL_SO = _HERE / "hc_split_sinkhorn_lib.so"

HC = 4
MIX_HC = (2 + HC) * HC  # 24
SINKHORN_ITERS = 20
EPS = 1e-6


def _sinkhorn(x: torch.Tensor, iters: int = SINKHORN_ITERS, eps: float = EPS):
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def hc_split_sinkhorn_ref(
    mixes: torch.Tensor,  # [n, MIX_HC] fp32
    hc_scale: torch.Tensor,  # [3]         fp32
    hc_base: torch.Tensor,  # [MIX_HC]    fp32
):
    n = mixes.shape[0]
    pre_in = mixes[:, :HC] * hc_scale[0] + hc_base[:HC]
    post_in = mixes[:, HC : 2 * HC] * hc_scale[1] + hc_base[HC : 2 * HC]
    comb_in = (mixes[:, 2 * HC :] * hc_scale[2] + hc_base[2 * HC :]).reshape(n, HC, HC)

    pre = torch.sigmoid(pre_in) + EPS
    post = 2.0 * torch.sigmoid(post_in)
    comb = _sinkhorn(comb_in)
    return pre, post, comb


_ARGTYPES = [
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int32,
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


def hc_split_sinkhorn(
    mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor
):
    """Run the device kernel, returning (pre, post, comb) entirely produced
    on-device."""
    assert mixes.is_npu and mixes.dtype == torch.float32
    assert hc_scale.is_npu and hc_scale.dtype == torch.float32
    assert hc_base.is_npu and hc_base.dtype == torch.float32
    assert mixes.dim() == 2 and mixes.shape[1] == MIX_HC
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (MIX_HC,)

    n = mixes.shape[0]
    mixes_c = mixes.contiguous()
    hc_scale_c = hc_scale.contiguous()
    hc_base_c = hc_base.contiguous()

    pre = torch.empty((n, HC), dtype=torch.float32, device=mixes.device)
    post = torch.empty((n, HC), dtype=torch.float32, device=mixes.device)
    comb = torch.empty((n, HC, HC), dtype=torch.float32, device=mixes.device)

    lib = _load()
    dev = torch.npu.current_device()
    blk = torch.npu.get_device_properties(dev).cube_core_num
    lib.call_kernel(
        blk,
        torch.npu.current_stream()._as_parameter_,
        ctypes.c_void_p(mixes_c.data_ptr()),
        ctypes.c_void_p(hc_scale_c.data_ptr()),
        ctypes.c_void_p(hc_base_c.data_ptr()),
        ctypes.c_void_p(pre.data_ptr()),
        ctypes.c_void_p(post.data_ptr()),
        ctypes.c_void_p(comb.data_ptr()),
        ctypes.c_int32(n),
    )
    torch.npu.synchronize()
    return pre, post, comb
