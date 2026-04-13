import os


DEVICE_ENV_VAR = "PTODSL_TEST_DEVICE_ID"
DEFAULT_DEVICE_ID = "0"
DEFAULT_NUM_CUBE_CORES = 20
DEFAULT_NUM_VEC_CORES = DEFAULT_NUM_CUBE_CORES * 2
DEVICE_PREFIX = "npu:"


# TODO: replace the torch-based device-property queries in get_num_cube_cores()
# and get_num_vec_cores() with the ACL C++ API (aclrtGetDeviceCount /
# aclrtGetDeviceName / aclrtDeviceGetInfo), the same way it is done in the
# pto-kernels host utilities:
#   https://github.com/huawei-csl/pto-kernels/blob/main/csrc/host/utils.h
# This removes the torch_npu dependency for pure-build / non-training workflows.


def get_num_cube_cores() -> int:
    """Return the number of cube (matrix) cores on the NPU."""
    try:
        import torch

        return int(getattr(torch.npu.get_device_properties(0), "cube_core_num"))
    except Exception as e:
        print(
            f"Warning: could not query cube_core_num ({e}); defaulting to {DEFAULT_NUM_CUBE_CORES}."
        )
        return DEFAULT_NUM_CUBE_CORES


def get_num_vec_cores() -> int:
    """Return the number of vector cores on the NPU.

    For all vector kernel tests: if bisheng is using mix-kernel compile mode,
    i.e. ``--npu-arch=dav-2201`` (which is the case for the tests here), the
    launch blockDim must be ``cube_cores`` (e.g. 24), not ``vec_cores`` (e.g.
    48).  The use of ``get_subblock_idx()`` will expand 24 → 48 logical workers
    automatically.

    Vector-only compile mode uses ``--cce-aicore-arch=dav-c220-vec`` (see the
    original C++ compile command); in that mode blockDim should be vec_cores.
    """
    try:
        import torch

        return int(getattr(torch.npu.get_device_properties(0), "vector_core_num"))
    except Exception as e:
        print(
            f"Warning: could not query vector_core_num ({e}); defaulting to {DEFAULT_NUM_VEC_CORES}."
        )
        return DEFAULT_NUM_VEC_CORES


def get_test_device() -> str:
    device_id = os.getenv(DEVICE_ENV_VAR)
    if not device_id:
        print(
            f"Warning: {DEVICE_ENV_VAR} is not set; defaulting to {DEFAULT_DEVICE_ID}."
        )
        device_id = DEFAULT_DEVICE_ID

    if device_id.startswith(DEVICE_PREFIX):
        return device_id
    return f"{DEVICE_PREFIX}{device_id}"
