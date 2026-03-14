import numpy as np


def swizzle_nz_py(li, m_loop, n_loop, c_swizzle, c1=1, c2=2):
    c_swizzle_m1 = c_swizzle - c1
    tile_block_loop = (n_loop + c_swizzle_m1) // c_swizzle
    tile_block_span = c_swizzle * m_loop
    tile_block_idx = li // tile_block_span
    in_tile_block_idx = li % tile_block_span
    is_last_block = tile_block_idx == (tile_block_loop - c1)
    n_col_tail = n_loop - c_swizzle * tile_block_idx
    n_col = n_col_tail if is_last_block else c_swizzle
    m_idx = in_tile_block_idx // n_col
    n_idx = tile_block_idx * c_swizzle + (in_tile_block_idx % n_col)
    odd_block = (tile_block_idx % c2) == c1
    if odd_block:
        m_idx = m_loop - m_idx - c1
    return m_idx, n_idx


def show_mapping(m_loop, n_loop, c_swizzle, preview=24):
    core_loop = m_loop * n_loop
    rows = []
    linear_order_grid = np.full((m_loop, n_loop), -1, dtype=np.int32)
    swizzle_order_grid = np.full((m_loop, n_loop), -1, dtype=np.int32)
    for li in range(min(core_loop, preview)):
        m_linear = li // n_loop
        n_linear = li % n_loop
        m_swz, n_swz = swizzle_nz_py(li, m_loop, n_loop, c_swizzle)
        linear_order_grid[m_linear, n_linear] = li
        swizzle_order_grid[m_swz, n_swz] = li
        rows.append((li, m_linear, n_linear, m_swz, n_swz))

    arr = np.array(rows, dtype=np.int32)
    print(
        f"\n=== swizzle={c_swizzle}, m_loop={m_loop}, n_loop={n_loop}, core_loop={core_loop} ==="
    )
    print("li | linear(m,n) -> swizzle(m,n)")
    for li, ml, nl, ms, ns in arr:
        print(f"{li:2d} | ({ml:2d},{nl:2d}) -> ({ms:2d},{ns:2d})")

    print("\nLinear traversal order grid (value = li):")
    print(linear_order_grid)
    print("\nSwizzled traversal order grid (value = li):")
    print(swizzle_order_grid)


if __name__ == "__main__":
    # Use a non-multiple n_loop to demonstrate tail handling.
    m_loop = 4
    n_loop = 7
    for c_swizzle in [2, 3, 5]:
        show_mapping(m_loop=m_loop, n_loop=n_loop, c_swizzle=c_swizzle, preview=28)
