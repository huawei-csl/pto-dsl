"""Run the deepseek_v4 ``hc_split_sinkhorn`` PTO kernel and validate
against the reference. Exits non-zero on mismatch."""

import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from ptodsl.npu_info import get_test_device

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from hc_split_sinkhorn_util import (  # noqa: E402
    MIX_HC,
    hc_split_sinkhorn,
    hc_split_sinkhorn_ref,
)


def main() -> int:
    device = get_test_device()
    torch.npu.set_device(device)
    torch.manual_seed(0)

    for n in (16, 64, 256, 1024):
        mixes = torch.randn(n, MIX_HC, dtype=torch.float32, device=device)
        hc_scale = torch.randn(3, dtype=torch.float32, device=device) * 0.5
        hc_base = torch.randn(MIX_HC, dtype=torch.float32, device=device) * 0.1

        pre_pto, post_pto, comb_pto = hc_split_sinkhorn(mixes, hc_scale, hc_base)
        pre_ref, post_ref, comb_ref = hc_split_sinkhorn_ref(mixes, hc_scale, hc_base)

        torch.testing.assert_close(pre_pto, pre_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(post_pto, post_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(comb_pto, comb_ref, rtol=1e-3, atol=1e-5)
        print(f"hc_split_sinkhorn n={n}: OK")
    print("hc_split_sinkhorn: all shapes PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
