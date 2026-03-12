from ptodsl import pto
from ptodsl import scalar as s

const = s.const

M_TILE = 128
K_QTILE = 64
K_TILE = 256
K_DTILE = 512
N_FULL = 256
SWIZZLE_COUNT = 5


def build_meta_data():
    def meta_data():
        dtype = pto.float16
        acc_dtype = pto.float32
        ptr_type = pto.PtrType(dtype)
        i32 = pto.int32
        tv_2d = pto.TensorType(rank=2, dtype=dtype)

        tile_view_a = pto.SubTensorType(shape=[M_TILE, K_DTILE], dtype=dtype)
        tile_view_b_256 = pto.SubTensorType(shape=[K_TILE, N_FULL], dtype=dtype)
        tile_view_c_256 = pto.SubTensorType(shape=[M_TILE, N_FULL], dtype=dtype)

        b_l1_cfg = pto.TileConfig(blayout="RowMajor", slayout="ColMajor", s_fractal_size=512)

        tile_buf_a_l1 = pto.TileType(shape=[M_TILE, K_DTILE], dtype=dtype, memory_space="MAT")
        tile_buf_b_l1_256 = pto.TileType(
            shape=[K_TILE, N_FULL], dtype=dtype, memory_space="MAT", config=b_l1_cfg
        )
        tile_buf_a_l0 = pto.TileType(shape=[M_TILE, K_QTILE], dtype=dtype, memory_space="LEFT")
        tile_buf_b_l0_256 = pto.TileType(shape=[K_QTILE, N_FULL], dtype=dtype, memory_space="RIGHT")
        tile_buf_c_256 = pto.TileType(shape=[M_TILE, N_FULL], dtype=acc_dtype, memory_space="ACC")

        return {
            "ptr_type": ptr_type,
            "i32": i32,
            "tv_2d": tv_2d,
            "tile_view_a": tile_view_a,
            "tile_view_b_256": tile_view_b_256,
            "tile_view_c_256": tile_view_c_256,
            "tile_buf_a_l1": tile_buf_a_l1,
            "tile_buf_b_l1_256": tile_buf_b_l1_256,
            "tile_buf_a_l0": tile_buf_a_l0,
            "tile_buf_b_l0_256": tile_buf_b_l0_256,
            "tile_buf_c_256": tile_buf_c_256,
        }

    return meta_data


def swizzle_nz(li, m_loop, n_loop, c_swizzle, c_swizzle_m1, c1, c2):
    tile_block_loop = (n_loop + c_swizzle_m1) // c_swizzle
    tile_block_span = c_swizzle * m_loop
    tile_block_idx = li // tile_block_span
    in_tile_block_idx = li % tile_block_span
    is_last_block = tile_block_idx == (tile_block_loop - c1)
    n_col_tail = n_loop - c_swizzle * tile_block_idx
    n_col = s.select(is_last_block, n_col_tail, c_swizzle)
    m_idx = in_tile_block_idx // n_col
    n_idx = tile_block_idx * c_swizzle + (in_tile_block_idx % n_col)
    odd_block = (tile_block_idx % c2) == c1
    flipped_m_idx = m_loop - m_idx - c1
    m_idx = s.select(odd_block, flipped_m_idx, m_idx)
    return m_idx, n_idx
