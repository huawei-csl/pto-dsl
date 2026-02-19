import ctypes
import torch
import torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)

    def matmul_func(
        c, a, b, batch_size,
        block_dim,
        stream_ptr=None
    ):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(c),
            torch_to_ctypes(a),
            torch_to_ctypes(b),
            ctypes.c_uint32(batch_size),
        )

    return matmul_func


def test_matmul(verbose=False):
    device = "npu:6"
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    m, k, n = 128, 128, 128
    batch_sizes = [5, 32, 66, 511]
    block_dims = [1, 2, 10, 20]
    matmul_func = load_lib("./matmul_kernel.so")

    torch.manual_seed(0)

    for bs in batch_sizes:
        a = torch.rand((bs, m, k), device=device, dtype=dtype)
        b = torch.rand((k, n), device=device, dtype=dtype)

        for block_dim in block_dims:
            c = torch.empty((bs, m, n), device=device, dtype=dtype)
            matmul_func(c, a, b, batch_size=a.shape[0], block_dim=block_dim)
            torch.npu.synchronize()

            c_ref = torch.matmul(a, b)
            diff = (c - c_ref).abs().max()
            print(f"config: bs={bs}, block_dim={block_dim}, max diff: {diff}")


if __name__ == "__main__":
    test_matmul()
