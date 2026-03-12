import numpy as np

M_TILE = 128
K_QTILE = 64
K_TILE = 256
K_DTILE = 512
N_FULL = 256


def step1_numpy_sim(a, b):
    """
    a: [m, k] float16/float32
    b: [n, k] float16/float32
    returns c: [m, n], equivalent to a @ b.T
    """
    m_total, k_total = a.shape
    n_total, k_total_b = b.shape
    assert k_total == k_total_b
    assert m_total % M_TILE == 0, "Step1 kernel uses full M tiles in this demo."
    assert k_total % K_DTILE == 0, "Step1 kernel uses full K_DTILE tiles."
    assert n_total % N_FULL == 0, "Tutorial simulation assumes full N tiles."

    # Corresponds to: n_loop, m_loop, core_loop, k_dtile_num
    n_loop = (n_total + N_FULL - 1) // N_FULL
    m_loop = m_total // M_TILE
    core_loop = n_loop * m_loop
    k_dtile_num = k_total // K_DTILE

    c = np.zeros((m_total, n_total), dtype=np.float32)

    # Corresponds to: for li in pto.range(...)
    for li in range(core_loop):
        # Corresponds to: m_idx = li // n_loop; n_idx = li % n_loop
        m_idx = li // n_loop
        n_idx = li % n_loop
        m_offset = m_idx * M_TILE
        n_offset = n_idx * N_FULL

        # Corresponds to tile accumulator c_l0
        c_tile = np.zeros((M_TILE, N_FULL), dtype=np.float32)

        # Corresponds to: load A tile once before k_idx loop
        a_l1 = a[m_offset : m_offset + M_TILE, 0:K_DTILE].astype(np.float32)

        for k_idx in range(k_dtile_num):
            k_offset = k_idx * K_DTILE
            is_first_k_tile = k_idx == 0

            # prefetch A tile for current k chunk (equivalent to pto.load)
            a_l1 = a[m_offset : m_offset + M_TILE, k_offset : k_offset + K_DTILE].astype(np.float32)

            # Corresponds to: for phase in range(8)
            for phase in range(8):
                # Corresponds to loading one B half tile every 4 phases
                if phase % 4 == 0:
                    b_half = phase // 4
                    h_off = b_half * K_TILE
                    b_l1 = b[n_offset : n_offset + N_FULL, k_offset + h_off : k_offset + h_off + K_TILE].astype(
                        np.float32
                    )

                # Corresponds to extract A/B quarter tiles
                a_col = phase * K_QTILE
                b_row = (phase % 4) * K_QTILE
                a_l0 = a_l1[:, a_col : a_col + K_QTILE]
                b_l0 = b_l1[:, b_row : b_row + K_QTILE]  # [N_FULL, K_QTILE]

                # Corresponds to matmul vs matmul_acc
                prod = a_l0 @ b_l0.T
                if phase == 0 and is_first_k_tile:
                    c_tile = prod
                else:
                    c_tile += prod

        c[m_offset : m_offset + M_TILE, n_offset : n_offset + N_FULL] = c_tile

    return c


def test_step1_numpy_sim():
    np.random.seed(0)
    for m, n, k in [(256, 512, 512), (384, 768, 1024)]:
        a = np.random.randn(m, k).astype(np.float16)
        b = np.random.randn(n, k).astype(np.float16)
        c_ref = a.astype(np.float32) @ b.astype(np.float32).T
        c_sim = step1_numpy_sim(a, b)
        np.testing.assert_allclose(c_sim, c_ref, rtol=1e-4, atol=1e-3)
    print("step1_numpy_sim unit test passed")


if __name__ == "__main__":
    test_step1_numpy_sim()
