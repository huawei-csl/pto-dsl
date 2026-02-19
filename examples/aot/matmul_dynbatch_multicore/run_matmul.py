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

    bs = 5
    m, k, n = 128, 128, 128
    torch.manual_seed(0)
    a = torch.rand((bs, m,k), device=device, dtype=dtype)
    b = torch.rand((k,n), device=device, dtype=dtype)
    c = torch.zeros((bs, m, n), device=device, dtype=dtype)

    matmul_func = load_lib("./matmul_kernel.so")
    matmul_func(c, a, b, batch_size=a.shape[0], block_dim=2)
    torch.npu.synchronize()

    c_ref = torch.matmul(a, b)
    diff = (c - c_ref).abs().max()
    print('max diff: ', diff)


    if verbose:
        print('ref')
        print(c_ref)
        print('our')
        print(c)
        tol = 1e-3
        correct = (c - c_ref).abs() <= tol

        batch_size, m_dim, n_dim = c.shape
        step_m = 4
        step_n = 4

        for bi in range(batch_size):
            print(f"\nBatch {bi}:")
            for i in range(0, m_dim, step_m):
                for j in range(0, n_dim, step_n):
                    if correct[bi, i : i + step_m, j : j + step_n].all():
                        print("X", end="")
                    else:
                        print(".", end="")
                print("|")


if __name__ == "__main__":
    test_matmul()
