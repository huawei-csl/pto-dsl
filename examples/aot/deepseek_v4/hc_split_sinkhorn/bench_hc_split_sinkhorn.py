"""Microbenchmark for the deepseek_v4 ``hc_split_sinkhorn`` PTO kernel.

Compares the fused on-device PTO kernel against the PyTorch reference
(pre/post sigmoid heads + 20-iter sinkhorn) over a sweep of batch
sizes ``n``.

Run::

    cd examples/aot/deepseek_v4/hc_split_sinkhorn
    bash compile.sh
    python bench_hc_split_sinkhorn.py
"""

import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from ptodsl import do_bench
from ptodsl.utils.npu_info import get_test_device

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hc_split_sinkhorn_util import (  # noqa: E402
    HC,
    MIX_HC,
    _KERNEL_SO,
    hc_split_sinkhorn,
    hc_split_sinkhorn_ref,
)


BATCHES = [64, 256, 1024, 4096, 16384]


def _alloc(n, device):
    torch.manual_seed(0)
    mixes = torch.randn(n, MIX_HC, dtype=torch.float32, device=device)
    hc_scale = torch.randn(3, dtype=torch.float32, device=device)
    hc_base = torch.randn(MIX_HC, dtype=torch.float32, device=device)
    return mixes, hc_scale, hc_base


def main():
    if not _KERNEL_SO.is_file():
        raise SystemExit(f"Build kernel first: cd {_HERE} && bash compile.sh")
    device = get_test_device()
    torch.npu.set_device(device)

    print(f"{'n':>7} {'pto us':>10} {'ref us':>10} {'speedup':>8}")
    print("-" * 40)
    for n in BATCHES:
        mixes, scale, base = _alloc(n, device)
        pto_us = do_bench(
            lambda: hc_split_sinkhorn(mixes, scale, base),
            warmup_iters=5,
            benchmark_iters=50,
            unit="us",
        )
        ref_us = do_bench(
            lambda: hc_split_sinkhorn_ref(mixes, scale, base),
            warmup_iters=5,
            benchmark_iters=50,
            unit="us",
        )
        speedup = ref_us / pto_us
        print(f"{n:>7} {pto_us:>10.2f} {ref_us:>10.2f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
