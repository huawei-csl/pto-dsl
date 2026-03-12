import numpy as np

M_TILE = 128
K_QTILE = 64
K_TILE = 256
K_DTILE = 512
N_FULL = 256


def _print_tile_memory(name, arr):
    kib = arr.nbytes / 1024
    print(f"[tile-mem] {name}: shape={arr.shape}, dtype={arr.dtype}, bytes={arr.nbytes} ({kib:.1f} KiB)")


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

    # Explicit tile-buffer allocation (mirrors pto.alloc_tile in step1_baseline.py).
    # Keep shapes fixed to tutorial constants for easy hardware-memory cross-checks.
    # a_l1: M_TILE * K_DTILE * sizeof(float16) = 128 * 512 * 2 = 131072 B = 128 KiB
    a_l1 = np.empty((M_TILE, K_DTILE), dtype=np.float16)
    # b_l1: K_TILE * N_FULL * sizeof(float16) = 256 * 256 * 2 = 131072 B = 128 KiB
    b_l1 = np.empty((K_TILE, N_FULL), dtype=np.float16)
    # a_l0: M_TILE * K_QTILE * sizeof(float16) = 128 * 64 * 2 = 16384 B = 16 KiB
    a_l0 = np.empty((M_TILE, K_QTILE), dtype=np.float16)
    # b_l0: K_QTILE * N_FULL * sizeof(float16) = 64 * 256 * 2 = 32768 B = 32 KiB
    b_l0 = np.empty((K_QTILE, N_FULL), dtype=np.float16)
    # c_tile: M_TILE * N_FULL * sizeof(float32) = 128 * 256 * 4 = 131072 B = 128 KiB
    c_tile = np.empty((M_TILE, N_FULL), dtype=np.float32)

    _print_tile_memory("a_l1", a_l1)
    _print_tile_memory("b_l1", b_l1)
    _print_tile_memory("a_l0", a_l0)
    _print_tile_memory("b_l0", b_l0)
    _print_tile_memory("c_tile", c_tile)

    # Corresponds to: for li in pto.range(...)
    for li in range(core_loop):
        # Corresponds to: m_idx = li // n_loop; n_idx = li % n_loop
        m_idx = li // n_loop
        n_idx = li % n_loop
        m_offset = m_idx * M_TILE
        n_offset = n_idx * N_FULL

        # Corresponds to tile accumulator c_l0 (reused buffer, reset per output tile).
        c_tile.fill(0.0)

        for k_idx in range(k_dtile_num):
            k_offset = k_idx * K_DTILE

            # Prefetch A tile for current K chunk (equivalent to pto.load into a_l1).
            a_l1[:, :] = a[m_offset : m_offset + M_TILE, k_offset : k_offset + K_DTILE]

            # Corresponds to: for phase in range(8)
            for phase in range(8):
                # Corresponds to loading one B half tile every 4 phases
                if phase % 4 == 0:
                    b_half = phase // 4
                    h_off = b_half * K_TILE
                    # b_l1 layout is [K_TILE, N_FULL], matching tile_buf_b_l1.
                    b_l1[:, :] = b[n_offset : n_offset + N_FULL, k_offset + h_off : k_offset + h_off + K_TILE].T

                # Corresponds to extract A/B quarter tiles
                a_col = phase * K_QTILE
                b_row = (phase % 4) * K_QTILE
                a_l0[:, :] = a_l1[:, a_col : a_col + K_QTILE]
                b_l0[:, :] = b_l1[b_row : b_row + K_QTILE, :]

                # Emulated tile matmul instruction:
                #   lhs a_l0: [M_TILE, K_QTILE] = [128, 64], fp16 source
                #   rhs b_l0: [K_QTILE, N_FULL] = [64, 256], fp16 source
                #   out c_tile: [M_TILE, N_FULL] = [128, 256], fp32 accumulate
                # Keep tile storage in fp16; cast only right at matmul for fp16->fp32 accumulate.
                c_tile += a_l0.astype(np.float32) @ b_l0.astype(np.float32)

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
