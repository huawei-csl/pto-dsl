import argparse

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

from common_utils import (
    K_DTILE,
    K_QTILE,
    K_TILE,
    M_TILE,
    N_FULL,
    build_meta_data,
    const,
)


def build():
    meta_data = build_meta_data()

    @to_ir_module(meta_data=meta_data)
    def matmul_kernel_ABt_autosync(
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

            tv_a = pto.as_tensor(tv_2d, ptr=a_ptr, shape=[m_total, k_total], strides=[k_total, c1])
            tv_b = pto.as_tensor(tv_2d, ptr=b_ptr, shape=[k_total, n_total], strides=[c1, k_total], layout="DN")
            tv_c = pto.as_tensor(tv_2d, ptr=c_ptr, shape=[m_total, n_total], strides=[n_total, c1])

            a_l1 = [pto.alloc_tile(tile_buf_a_l1), pto.alloc_tile(tile_buf_a_l1)]
            b_l1 = [pto.alloc_tile(tile_buf_b_l1), pto.alloc_tile(tile_buf_b_l1)]
            a_l0 = [pto.alloc_tile(tile_buf_a_l0), pto.alloc_tile(tile_buf_a_l0)]
            b_l0 = [pto.alloc_tile(tile_buf_b_l0), pto.alloc_tile(tile_buf_b_l0)]
            c_l0 = pto.alloc_tile(tile_buf_c)

            for li in pto.range(bid, core_loop, num_blocks):
                m_idx = li // n_loop
                n_idx = li % n_loop

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
                pto.load(sv_a0, a_l1[0])

                for k_idx in pto.range(c0, k_dtile_num, c1):
                    k_offset = k_idx * c_kd

                    def run_loop_k(a_curr, a_next):
                        is_first_k_tile = k_idx == c0

                        for h in range(2):
                            h_off = const(h * K_TILE)
                            sv_b = pto.slice_view(
                                tile_view_b,
                                source=tv_b,
                                offsets=[k_offset + h_off, n_offset],
                                sizes=[c_kt, c_nt],
                            )
                            pto.load(sv_b, b_l1[h])

                            for quarter in range(4):
                                phase = h * 4 + quarter
                                ping = phase & 1
                                a_col = const(phase * K_QTILE)
                                b_row = const(quarter * K_QTILE)

                                tile.extract(a_curr, c0, a_col, a_l0[ping])
                                tile.extract(b_l1[h], b_row, c0, b_l0[ping])

                                if phase == 0:
                                    pto.cond(
                                        is_first_k_tile,
                                        lambda: tile.matmul(a_l0[ping], b_l0[ping], c_l0),
                                        lambda: tile.matmul_acc(c_l0, a_l0[ping], b_l0[ping], c_l0),
                                    )
                                else:
                                    tile.matmul_acc(c_l0, a_l0[ping], b_l0[ping], c_l0)

                        with pto.if_context(k_idx + c1 < k_dtile_num):
                            sv_a_next = pto.slice_view(
                                tile_view_a,
                                source=tv_a,
                                offsets=[m_offset, k_offset + c_kd],
                                sizes=[const(M_TILE), c_kd],
                            )
                            pto.load(sv_a_next, a_next)

                    is_curr0 = (k_idx % c2) == c0
                    with pto.if_context(is_curr0, has_else=True) as branch:
                        run_loop_k(a_l1[0], a_l1[1])
                    with branch.else_context():
                        run_loop_k(a_l1[1], a_l1[0])

                sv_c = pto.slice_view(
                    tile_view_c,
                    source=tv_c,
                    offsets=[m_offset, n_offset],
                    sizes=[const(M_TILE), c_nt],
                )
                pto.store(c_l0, sv_c)

    return matmul_kernel_ABt_autosync


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    print(build())
