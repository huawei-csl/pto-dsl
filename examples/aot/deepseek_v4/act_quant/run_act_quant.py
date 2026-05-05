"""Run the deepseek_v4 ``act_quant`` PTO kernel and validate against the
reference. Exits non-zero on mismatch."""

import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from ptodsl.npu_info import get_test_device

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from act_quant_util import BLOCK_SIZE, act_quant, act_quant_ref  # noqa: E402


def main() -> int:
    device = get_test_device()
    torch.npu.set_device(device)
    torch.manual_seed(0)

    shapes = [
        (32, BLOCK_SIZE),
        (64, BLOCK_SIZE * 2),
        (128, BLOCK_SIZE * 4),
    ]
    for M, N in shapes:
        x = torch.randn(M, N, dtype=torch.float16, device=device)
        y_pto, s_pto = act_quant(x)
        y_ref, s_ref = act_quant_ref(x)
        torch.testing.assert_close(s_pto, s_ref, rtol=1e-3, atol=1e-6)
        diff = (y_pto.to(torch.int32) - y_ref.to(torch.int32)).abs()
        max_diff = diff.max().item()
        match = (diff == 0).float().mean().item()
        assert max_diff <= 1, f"M={M} N={N}: max int8 diff = {max_diff}"
        assert match > 0.95, f"M={M} N={N}: only {match * 100:.1f}% exact"
        print(f"act_quant M={M} N={N}: max_diff={max_diff} exact={match * 100:.1f}% OK")
    print("act_quant: all shapes PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
