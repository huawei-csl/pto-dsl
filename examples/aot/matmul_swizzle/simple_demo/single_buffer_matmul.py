import argparse

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const


def build():
    M_TILE = 128
    K_QTILE = 64
    K_TILE = 256
    K_DTILE = 512
    N_FULL = 256
    SWIZZLE_COUNT = 5

    def meta_data():
        dtype = pto.float16
        acc_dtype = pto.float32
        ptr_type = pto.PtrType(dtype)
        i32 = pto.int32
        tv_2d = pto.TensorType(rank=2, dtype=dtype)

        tile_view_a = pto.SubTensorType(shape=[M_TILE, K_DTILE], dtype=dtype)
        tile_view_b_256 = pto.SubTensorType(shape=[K_TILE, N_FULL], dtype=dtype)
        tile_view_c_256 = pto.SubTensorType(shape=[M_TILE, N_FULL], dtype=dtype)

        b_l1_cfg = pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor", s_fractal_size=512)

        tile_buf_a_l1 = pto.TileBufType(shape=[M_TILE, K_DTILE], dtype=dtype, memory_space="MAT")
        tile_buf_b_l1_256 = pto.TileBufType(
            shape=[K_TILE, N_FULL], dtype=dtype, memory_space="MAT", config=b_l1_cfg
        )
        tile_buf_a_l0 = pto.TileBufType(shape=[M_TILE, K_QTILE], dtype=dtype, memory_space="LEFT")
        tile_buf_b_l0_256 = pto.TileBufType(shape=[K_QTILE, N_FULL], dtype=dtype, memory_space="RIGHT")
        tile_buf_c_256 = pto.TileBufType(shape=[M_TILE, N_FULL], dtype=acc_dtype, memory_space="ACC")

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

    @to_ir_module(meta_data=meta_data)
    def matmul_kernel_ABt_single_buffer_autosync(
        a_ptr: "ptr_type",
        b_ptr: "ptr_type",
        c_ptr: "ptr_type",
        m_i32: "i32",
        n_i32: "i32",
        k_i32: "i32",
    ) -> None:
        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            c2 = const(2)
            c128 = const(M_TILE)
            c256 = const(N_FULL)
            c512 = const(K_DTILE)

            m_total = s.index_cast(m_i32)
            n_total = s.index_cast(n_i32)
            k_total = s.index_cast(k_i32)
            num_blocks = s.index_cast(pto.get_block_num())
            bid = s.index_cast(pto.get_block_idx())

            n_loop = (n_total + c256 - c1) // c256
            m_loop = m_total // c128
            core_loop = n_loop * m_loop
            k_dtile_num = k_total // c512
            c_swizzle = const(SWIZZLE_COUNT)
            c_swizzle_m1 = c_swizzle - c1

            tv_a = pto.as_tensor(tv_2d, ptr=a_ptr, shape=[m_total, k_total], strides=[k_total, c1])
            tv_b = pto.as_tensor(tv_2d, ptr=b_ptr, shape=[k_total, n_total], strides=[c1, k_total], layout="DN")
            tv_c = pto.as_tensor(tv_2d, ptr=c_ptr, shape=[m_total, n_total], strides=[n_total, c1])

            a_l1 = pto.alloc_tile(tile_buf_a_l1)
            b_l1 = pto.alloc_tile(tile_buf_b_l1_256)
            a_l0 = pto.alloc_tile(tile_buf_a_l0)
            b_l0 = pto.alloc_tile(tile_buf_b_l0_256)
            c_l0 = pto.alloc_tile(tile_buf_c_256)

            for li in pto.range(bid, core_loop, num_blocks):
                m_idx, n_idx = swizzle_nz(li, m_loop, n_loop, c_swizzle, c_swizzle_m1, c1, c2)
                m_offset = m_idx * c128
                n_offset = n_idx * c256
                c_kt = const(K_TILE)
                c_kd = const(K_DTILE)
                c_nt = const(N_FULL)

                sv_a0 = pto.slice_view(
                    tile_view_a,
                    source=tv_a,
                    offsets=[m_offset, c0],
                    sizes=[const(M_TILE), c_kd],
                )
                pto.load(sv_a0, a_l1)

                for k_idx in pto.range(c0, k_dtile_num, c1):
                    k_offset = k_idx * c_kd
                    is_first_k_tile = k_idx == c0

                    for h in range(2):
                        h_off = const(h * K_TILE)
                        sv_b = pto.slice_view(
                            tile_view_b_256,
                            source=tv_b,
                            offsets=[k_offset + h_off, n_offset],
                            sizes=[c_kt, c_nt],
                        )
                        pto.load(sv_b, b_l1)

                        for quarter in range(4):
                            phase = h * 4 + quarter
                            a_col = const(phase * K_QTILE)
                            b_row = const(quarter * K_QTILE)

                            tile.extract(a_l1, c0, a_col, a_l0)
                            tile.extract(b_l1, b_row, c0, b_l0)

                            if phase == 0:
                                pto.cond(
                                    is_first_k_tile,
                                    lambda: tile.matmul(a_l0, b_l0, c_l0),
                                    lambda: tile.matmul_acc(c_l0, a_l0, b_l0, c_l0),
                                )
                            else:
                                tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)

                    with pto.if_context(k_idx + c1 < k_dtile_num):
                        sv_a_next = pto.slice_view(
                            tile_view_a,
                            source=tv_a,
                            offsets=[m_offset, k_offset + c_kd],
                            sizes=[const(M_TILE), c_kd],
                        )
                        pto.load(sv_a_next, a_l1)

                sv_c = pto.slice_view(
                    tile_view_c_256,
                    source=tv_c,
                    offsets=[m_offset, n_offset],
                    sizes=[const(M_TILE), c_nt],
                )
                pto.store(c_l0, sv_c)

    return matmul_kernel_ABt_single_buffer_autosync


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    print(build())
