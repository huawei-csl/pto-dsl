# Copyright (c) Tile-AI Corporation.
# SPDX-License-Identifier: MIT
"""TileLang NPU kernel: Sinkhorn normalization (forward-only).

Algorithm matches ``sinkhorn_normalize_ref`` in this file: stable softmax on the
last axis, add ``eps``, then alternate row/column normalization for
``sinkhorn_iters`` total steps (one column step after softmax, then pairs of
row/column steps). The TileLang body uses shared-tile reductions and elementwise
``T.tile`` ops analogous to a hand-written Sinkhorn loop on an ``(hc, hc)`` tile.
"""

from __future__ import annotations

import tilelang
import torch
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def sinkhorn_normalize_ref(
    x: torch.Tensor, repeat: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    """Reference Sinkhorn forward (PyTorch)."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def sinkhorn_normalize_kernel(hc: int, sinkhorn_iters: int, eps: float):
    """JIT-compiled kernel: ``out[n, hc, hc] = sinkhorn(inp[n, hc, hc])``."""
    n = T.symbolic("n")
    dtype = "float"

    block_M = 2
    vec_num = 2
    hc_pad = hc
    if hc * 4 % 32 != 0:
        hc_pad = tilelang.cdiv(hc * 4, 32) * 32 // 4

    m_num = tilelang.cdiv(n, block_M)

    @T.prim_func
    def main(
        inp: T.Tensor([n, hc, hc], dtype),
        out: T.Tensor([n, hc, hc], dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            comb_shared = T.alloc_shared((hc, hc_pad), dtype)
            tmp_shared = T.alloc_shared(hc_pad, dtype)
            row_sum = T.alloc_shared(hc_pad, dtype)
            col_sum = T.alloc_shared(hc_pad, dtype)
            row_max = T.alloc_shared(hc_pad, dtype)

            gid = cid * block_M + vid * block_M // vec_num
            if gid < n:
                for i in T.serial(hc):
                    T.copy(inp[gid, i, :], tmp_shared)
                    T.copy(tmp_shared, comb_shared[i, :])

                # comb = comb.softmax(-1) + eps
                T.reduce_max(comb_shared, row_max, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.sub(comb_shared[i, :], comb_shared[i, :], row_max[i])
                T.tile.exp(comb_shared, comb_shared)
                T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                for i in T.serial(hc):
                    T.tile.div(comb_shared[i, :], comb_shared[i, :], row_sum[i])
                T.tile.add(comb_shared, comb_shared, eps)

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                T.tile.add(col_sum, col_sum, eps)
                for i in T.serial(hc):
                    T.tile.div(comb_shared[i, :], comb_shared[i, :], col_sum)

                for _ in T.serial(sinkhorn_iters - 1):
                    T.reduce_sum(comb_shared, row_sum, dim=-1, real_shape=[hc, hc])
                    T.tile.add(row_sum, row_sum, eps)
                    for i in T.serial(hc):
                        T.tile.div(comb_shared[i, :], comb_shared[i, :], row_sum[i])
                    T.reduce_sum(comb_shared, col_sum, dim=0, real_shape=[hc, hc_pad])
                    T.tile.add(col_sum, col_sum, eps)
                    for i in T.serial(hc):
                        T.tile.div(comb_shared[i, :], comb_shared[i, :], col_sum)

                for i in T.serial(hc):
                    T.copy(comb_shared[i, :hc], out[gid, i, :])

    return main


def build_sinkhorn(hc: int = 4, sinkhorn_iters: int = 10, eps: float = 1e-6):
    """Return compiled callable ``fn(inp) -> out`` (``inp`` is NPU float32)."""
    return sinkhorn_normalize_kernel(hc, sinkhorn_iters, eps)


# ``m_num = ceil(n / block_M)`` with ``block_M = 2`` must stay below the Ascend
# launch grid limit (~65535); two matrices are scheduled per ``m_num`` slot
# (``vec_num = 2``), so ``n <= 2 * 65535`` is safe for one launch.
_MAX_N_SINGLE_LAUNCH = 2 * 65535


def sinkhorn_normalize_tilelang(
    inp: torch.Tensor,
    *,
    hc: int,
    sinkhorn_iters: int = 10,
    eps: float = 1e-6,
    _fn=None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Sinkhorn on ``inp`` shaped ``[n, hc, hc]`` (float32 on NPU).

    Large ``n`` are split into multiple kernel launches to respect grid limits.
    """
    if inp.dim() != 3 or inp.shape[1] != hc or inp.shape[2] != hc:
        raise ValueError(f"expected [n, {hc}, {hc}], got {tuple(inp.shape)}")
    n = inp.shape[0]
    if out is None:
        out = torch.empty_like(inp)
    elif out.shape != inp.shape or out.dtype != inp.dtype:
        raise ValueError("out must match inp shape and dtype")
    fn = _fn or build_sinkhorn(hc, sinkhorn_iters, eps)
    off = 0
    while off < n:
        m = min(_MAX_N_SINGLE_LAUNCH, n - off)
        sl = slice(off, off + m)
        out[sl] = fn(inp[sl])
        off += m
    return out
