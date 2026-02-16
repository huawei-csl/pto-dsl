import ctypes
import torch
import torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def lib_to_func(lib):
    def add_func(
        x,
        y,
        z,
        stream_ptr=None
        ):

        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_

        N = x.numel()
        lib.call_kernel(
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            torch_to_ctypes(z),
            N
        )
    return add_func


def test_add():
    device = "npu:7"
    torch.npu.set_device(device)

    lib_path = "./add_lib.so"
    lib = ctypes.CDLL(lib_path)
    add_func = lib_to_func(lib)

    # shape parameter hard-coded as kernel
    num_cores = 20 * 2
    tile_size = 1024
    # Use general (non tile-aligned) sizes to exercise tail tiles and
    # cross-core partitioning together.
    shape_list = [
        1,
        17,
        tile_size - 1,
        tile_size + 13,
        num_cores * tile_size - 3,
        num_cores * tile_size + 29,
        2 * num_cores * tile_size + 7,
        5 * num_cores * tile_size - 11,
    ]

    torch.manual_seed(0)
    dtype = torch.float32

    for shape in shape_list:
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)

        add_func(x, y, z)
        torch.npu.synchronize()

        z_ref = x + y
        try:
            torch.testing.assert_close(z, z_ref)
            print(f"result equal for shape {shape}")
        except AssertionError as e:
            print(f"result mismatch for shape {shape}: {e}")

if __name__ == "__main__":
    test_add()
