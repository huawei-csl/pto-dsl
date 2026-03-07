import sys

from .api import pto, scalar, tile
from .compiler.ir import to_ir_module
from .compiler.jit import JitWrapper, jit
from .utils import bench, test_util
from .utils.bench import do_bench

sys.modules[__name__ + ".pto"] = pto
sys.modules[__name__ + ".scalar"] = scalar
sys.modules[__name__ + ".tile"] = tile
sys.modules[__name__ + ".bench"] = bench
sys.modules[__name__ + ".test_util"] = test_util

__all__ = ["JitWrapper", "do_bench", "jit", "pto", "scalar", "tile", "to_ir_module"]
