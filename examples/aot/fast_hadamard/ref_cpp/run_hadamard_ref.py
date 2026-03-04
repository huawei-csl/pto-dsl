import ctypes

import torch
import torch_npu
import math

BLOCK_DIM = 24

def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)

    # call_kernel(blockDim, stream, x, batch, n, log2_n)
    lib.call_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # x (in-place)
        ctypes.c_uint32,  # batch
        ctypes.c_uint32,  # n
        ctypes.c_uint32,  # log2_n
    ]
    lib.call_kernel.restype = None

    default_block_dim = BLOCK_DIM

    def hadamard_func(
        x, batch, n, log2_n, block_dim=default_block_dim, stream_ptr=None
    ):
        if stream_ptr is None:
            stream = torch.npu.current_stream()
            stream_ptr = getattr(stream, "_as_parameter_", None)
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            batch,
            n,
            log2_n,
        )

    return hadamard_func


def hadamard_ref_inplace(x):
    """Reference FHT matching TGATHER(P0101/P1010) + TADD/TSUB layout."""
    x = x.clone()
    n = x.shape[-1]
    n_half = n // 2
    log2_n = int(math.log2(n))
    for _ in range(log2_n):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd
    return x


def _is_power_of_two(v):
    return v > 0 and (v & (v - 1)) == 0


def test_correctness(hadamard_func):
    """Run correctness tests across (batch, N, seed) configs."""
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)

    TEST_BATCHES = [1, 7, 22, 65]
    TEST_HIDDEN_DIMS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    TEST_SEEDS = [0, 1]

    passed = 0
    total = 0
    for seed in TEST_SEEDS:
        for batch in TEST_BATCHES:
            for n in TEST_HIDDEN_DIMS:
                total += 1
                torch.manual_seed(seed)
                log2_n = int(math.log2(n))
                x = torch.randn(batch, n, device="npu", dtype=torch.float16)

                y_ref = hadamard_ref_inplace(x)

                hadamard_func(x, batch, n, log2_n)
                torch.npu.synchronize()

                if torch.equal(x, y_ref):
                    passed += 1
                    print(f"  PASS  seed={seed} batch={batch:>4d}, N={n:>5d}")
                else:
                    maxdiff = (x - y_ref).abs().max().item()
                    print(
                        f"  FAIL  seed={seed} batch={batch:>4d}, N={n:>5d}"
                        f"  max_diff={maxdiff:.6f}"
                    )

    print(f"\n{passed}/{total} tests passed.\n")



if __name__ == "__main__":
    device = "npu:7"
    torch.npu.set_device(device)

    hadamard_func = load_lib(lib_path="./fast_hadamard_ref_lib.so")
    test_correctness(hadamard_func)
