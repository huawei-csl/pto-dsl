"""Sinkhorn normalization: PTO kernel vs PyTorch reference."""

import argparse
import ctypes

import torch
import torch_npu  # noqa: F401

from ptodsl.npu_info import get_num_cube_cores, get_test_device

_DEFAULT_NUM_CORES = get_num_cube_cores()
ROW_CHUNK = 8


def torch_to_ctypes(t):
    return ctypes.c_void_p(t.data_ptr())


def load_lib(lib_path, block_dim=_DEFAULT_NUM_CORES):
    lib = ctypes.CDLL(lib_path)
    lib.call_sinkhorn_kernel.argtypes = [
        ctypes.c_uint32,  # blockDim
        ctypes.c_void_p,  # stream
        ctypes.c_void_p,  # matrix_in
        ctypes.c_void_p,  # matrix_out
        ctypes.c_void_p,  # mu1_out
        ctypes.c_void_p,  # mu2_out
        ctypes.c_uint32,  # N
        ctypes.c_uint32,  # K
        ctypes.c_uint32,  # L
        ctypes.c_uint32,  # order
        ctypes.c_float,  # lr
        ctypes.c_float,  # eps
        ctypes.c_float,  # invK
        ctypes.c_float,  # invL
        ctypes.c_float,  # invK1
        ctypes.c_float,  # invL1
    ]
    lib.call_sinkhorn_kernel.restype = None

    def sinkhorn(
        mat_in,
        mat_out,
        mu1_out,
        mu2_out,
        N,
        K,
        L,
        order,
        lr,
        eps,
        block_dim=block_dim,
        stream_ptr=None,
    ):
        if stream_ptr is None:
            stream_ptr = torch.npu.current_stream()._as_parameter_
        invK = 1.0 / K
        invL = 1.0 / L
        invK1 = 1.0 / max(K - 1, 1)
        invL1 = 1.0 / max(L - 1, 1)
        lib.call_sinkhorn_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(mat_in),
            torch_to_ctypes(mat_out),
            torch_to_ctypes(mu1_out),
            torch_to_ctypes(mu2_out),
            N,
            K,
            L,
            order,
            lr,
            eps,
            invK,
            invL,
            invK1,
            invL1,
        )

    return sinkhorn


def sinkhorn_ref(matrix_in, order, lr, eps):
    """PyTorch reference Sinkhorn normalization (matches reference.cpp).

    Per (K, L) matrix:
      mu1[L] = mu2[K] = 1
      For each phase 0..order:
        cm = matrix_in / (mu2[:, None] * mu1[None, :])
        rStd[k] = unbiased std of cm[k, :]
        cStd[l] = unbiased std of cm[:, l]
        if phase == 0: tgt = min(rStd.min(), cStd.min()) + eps
        else:          mu2 *= (rStd / tgt) ** lr
                       mu1 *= (cStd / tgt) ** lr
      out = matrix_in / (mu2[:, None] * mu1[None, :])
    """
    N, K, L = matrix_in.shape
    cm_in = matrix_in.float()
    out = torch.empty_like(cm_in)
    mu1_all = torch.empty(N, L, device=matrix_in.device, dtype=torch.float32)
    mu2_all = torch.empty(N, K, device=matrix_in.device, dtype=torch.float32)

    for bi in range(N):
        cm0 = cm_in[bi]
        mu1 = torch.ones(L, device=matrix_in.device, dtype=torch.float32)
        mu2 = torch.ones(K, device=matrix_in.device, dtype=torch.float32)
        tgt = None
        for phase in range(order + 1):
            cm = cm0 / (mu2[:, None] * mu1[None, :])
            rStd = cm.std(dim=1, unbiased=True)
            cStd = cm.std(dim=0, unbiased=True)
            if phase == 0:
                tgt = torch.minimum(rStd.min(), cStd.min()) + eps
            else:
                mu2 = mu2 * torch.clamp(rStd / tgt, min=1e-12).pow(lr)
                mu1 = mu1 * torch.clamp(cStd / tgt, min=1e-12).pow(lr)
        out[bi] = cm0 / (mu2[:, None] * mu1[None, :])
        mu1_all[bi] = mu1
        mu2_all[bi] = mu2
    return (
        out.to(matrix_in.dtype),
        mu1_all.to(matrix_in.dtype),
        mu2_all.to(matrix_in.dtype),
    )


def test_sinkhorn(lib_path, block_dim=_DEFAULT_NUM_CORES):
    device = get_test_device()
    torch.npu.set_device(device)

    sinkhorn = load_lib(lib_path=lib_path, block_dim=block_dim)

    torch.manual_seed(0)
    dtype = torch.float16
    # Mirrors the upstream torch_npu test suite (shapes x orders x seeds);
    # cases where K is not a multiple of ROW_CHUNK=8 are skipped because the
    # current builder only supports K-aligned chunking.
    SHAPES = [
        (1, 16, 16),
        (1, 32, 32),
        (1, 64, 64),
        (1, 128, 128),
        (1, 256, 256),
        (2, 64, 64),
        (4, 32, 64),
        (4, 64, 32),
        (8, 128, 128),
        (1, 16, 256),
        (1, 256, 16),
    ]
    ORDERS = [1, 5, 10]
    SEEDS = [0, 42]
    LR, EPS = 0.5, 1e-3
    cases = [
        (N, K, L, order, LR, EPS, seed)
        for (N, K, L) in SHAPES
        for order in ORDERS
        for seed in SEEDS
    ]

    results = []
    for N, K, L, order, lr, eps, seed in cases:
        if K % ROW_CHUNK != 0:
            print(
                f"[skip ] N={N} K={K} L={L} order={order} seed={seed} "
                f"(K not multiple of {ROW_CHUNK})"
            )
            results.append((N, K, L, order, seed, "skip"))
            continue
        torch.manual_seed(seed)
        # Positive entries: keep within fp16 range.
        mat_in = torch.rand(N, K, L, device=device, dtype=dtype) + 0.1
        mat_out = torch.empty_like(mat_in)
        mu1_out = torch.empty(N, L, device=device, dtype=dtype)
        mu2_out = torch.empty(N, K, device=device, dtype=dtype)

        ref_out, ref_mu1, ref_mu2 = sinkhorn_ref(mat_in, order, lr, eps)

        sinkhorn(mat_in, mat_out, mu1_out, mu2_out, N, K, L, order, lr, eps)
        torch.npu.synchronize()

        ok = True
        details = []
        for name, got, want in [
            ("matrix_out", mat_out, ref_out),
            ("mu1_out", mu1_out, ref_mu1),
            ("mu2_out", mu2_out, ref_mu2),
        ]:
            try:
                torch.testing.assert_close(got, want, rtol=5e-2, atol=1e-2)
            except AssertionError as err:
                ok = False
                details.append(f"  {name}: {str(err).strip()[:200]}")

        status = "match" if ok else "mismatch"
        print(
            f"[{status}] N={N} K={K} L={L} order={order} seed={seed} "
            f"lr={lr} eps={eps}"
        )
        for d in details:
            print(d)
        results.append((N, K, L, order, seed, status))

    print("\nsummary:")
    counts = {"match": 0, "mismatch": 0, "skip": 0}
    for r in results:
        counts[r[-1]] = counts.get(r[-1], 0) + 1
        print("  ", r)
    print(f"\n  totals: {counts}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="./sinkhorn_lib.so")
    parser.add_argument("--block-dim", type=int, default=_DEFAULT_NUM_CORES)
    args = parser.parse_args()
    test_sinkhorn(args.lib, block_dim=args.block_dim)
