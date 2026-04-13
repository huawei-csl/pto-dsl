from .bench import do_bench
from .npu_info import get_num_cube_cores, get_num_vec_cores, get_test_device

__all__ = ["do_bench", "get_num_cube_cores", "get_num_vec_cores", "get_test_device"]
