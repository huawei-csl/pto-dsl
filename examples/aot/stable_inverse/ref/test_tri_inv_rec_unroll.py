# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------


import torch
import pytest
import numpy as np
import random
from typing import Callable
from pto_kernels import pto_tri_inv_rec_unroll

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.1):
    U = scale * torch.triu(torch.rand((block_dim_x, block_dim_y, n, n)), diagonal=1)
    return U


def ones_triu_matrix(n, block_dim_x, block_dim_y):
    U = torch.triu(torch.ones((block_dim_x, block_dim_y, n, n)), diagonal=1)
    return U


def block_ones_triu_matrix(n, block_dim_x, block_dim_y):
    U_ = np.ones((16, 16))
    n_blocks = n // 16
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(n_blocks):
                start = i * 16
                end = i * 16 + 16
                U[x, y, start:end, start:end] = U_
    return torch.from_numpy(np.triu(U, 1))


def block_random_triu_matrix(n, block_dim_x, block_dim_y, scale=0.2):
    U_ = scale * np.random.rand(16, 16)
    U_ = np.triu(U_, k=1)
    U = np.zeros((block_dim_x, block_dim_y, n, n))
    for x in range(block_dim_x):
        for y in range(block_dim_y):
            for i in range(0, n, 16):
                U[x, y, i : i + 16, i : i + 16] = U_.copy()
    return torch.from_numpy(U)


def linalg_inv(U: torch.tensor) -> torch.tensor:
    n = U.shape[-1]
    Identity = np.ones((n, n), dtype=np.double)
    Identity = np.triu(Identity)
    Identity = np.tril(Identity)
    golden_numpy = np.zeros((U.shape))
    for x in range(U.shape[0]):
        for y in range(U.shape[1]):
            golden_numpy[x, y] = np.linalg.inv(
                U[x, y].numpy().astype(np.double) + Identity
            )
    return torch.from_numpy(golden_numpy)


def _test_tri_inv_rec_unroll(U: torch.tensor, atol: float, rtol: float, ftol: float):

    U = U.to(torch.half)
    golden_cpu = linalg_inv(U)

    U_npu = U.npu()

    torch.npu.synchronize()
    actual = pto_tri_inv_rec_unroll(U_npu, is_bsnd_format=False)
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    torch.npu.synchronize()
    actual_cpu = actual_cpu.to(torch.float64)
    frob_error = torch.sqrt(
        torch.sum((golden_cpu - actual_cpu) * (golden_cpu - actual_cpu))
        / torch.sum(golden_cpu * golden_cpu)
    )
    actual_numpy = actual_cpu.numpy()
    golden_numpy = golden_cpu.numpy()

    assert np.allclose(
        actual_numpy, golden_numpy, atol=atol, rtol=rtol
    ), f"Error at allclose - tensor shape: {U.shape} - rtol: {rtol}."
    assert frob_error <= ftol, f"frob_error: {frob_error}"


def _test_tri_inv_rec_unroll_bsnd(
    U: torch.tensor,
    B: int,
    S: int,
    N: int,
    D: int,
    atol: float,
    rtol: float,
    ftol: float,
):

    U = U.to(torch.half)
    golden_cpu = linalg_inv(U)

    # Transform to bsnd layout
    U = U.transpose(1, 2).contiguous().reshape(B, S, N, D)
    torch.npu.synchronize()
    golden_cpu = golden_cpu.transpose(1, 2).contiguous().reshape(B, S, N, D)

    U_npu = U.npu()

    torch.npu.synchronize()
    actual = pto_tri_inv_rec_unroll(U_npu, is_bsnd_format=True)
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    torch.npu.synchronize()
    actual_cpu = actual_cpu.to(torch.float64)
    frob_error = torch.sqrt(
        torch.sum((golden_cpu - actual_cpu) * (golden_cpu - actual_cpu))
        / torch.sum(golden_cpu * golden_cpu)
    )
    actual_numpy = actual_cpu.numpy()
    golden_numpy = golden_cpu.numpy()

    assert np.allclose(
        actual_numpy, golden_numpy, atol=atol, rtol=rtol
    ), f"Error at allclose - tensor shape: {U.shape} - rtol: {rtol}."
    assert frob_error <= ftol, f"frob_error: {frob_error}"


@pytest.mark.parametrize("n", [16, 32, 64, 128])
@pytest.mark.parametrize("block_dim_x", [1, 2, 3, 4])
@pytest.mark.parametrize("block_dim_y", [2, 4, 8])
@pytest.mark.parametrize(
    "matrix_gen,atol,rtol,ftol",
    [
        (block_ones_triu_matrix, 0, 0, 0),
        (ones_triu_matrix, 0, 0, 0),
        (block_random_triu_matrix, 5e-5, 0.1, 1e-4),
        (random_triu_matrix, 5e-5, 0.1, 1e-4),
    ],
)
def test_tri_inv_rec_unroll(
    n: int,
    block_dim_x: int,
    block_dim_y: int,
    matrix_gen: Callable,
    atol: float,
    rtol: float,
    ftol: float,
):
    U = matrix_gen(n, block_dim_x, block_dim_y)
    _test_tri_inv_rec_unroll(U, atol, rtol, ftol)


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S", [128, 256, 1024])
@pytest.mark.parametrize("N", [4, 8])
@pytest.mark.parametrize("D", [16, 32, 64, 128])
@pytest.mark.parametrize(
    "matrix_gen,atol,rtol,ftol",
    [
        (block_ones_triu_matrix, 0, 0, 0),
        (ones_triu_matrix, 0, 0, 0),
        (block_random_triu_matrix, 5e-5, 0.1, 1e-4),
        (random_triu_matrix, 5e-5, 0.1, 1e-4),
    ],
)
def test_tri_inv_rec_unroll_bsnd(
    B: int,
    S: int,
    N: int,
    D: int,
    matrix_gen: Callable,
    atol: float,
    rtol: float,
    ftol: float,
):
    # only test cases where the sequence length is a multiple of the chunk size are accepted
    if S % D != 0:
        pytest.skip("Sequence length must be a multiple of chunk size D.")
    U = matrix_gen(D, B * S // D, N)
    _test_tri_inv_rec_unroll_bsnd(U, B, S, N, D, atol, rtol, ftol)
