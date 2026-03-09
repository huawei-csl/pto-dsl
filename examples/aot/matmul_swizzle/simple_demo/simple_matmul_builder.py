from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

def build():
    M_TILE = 128
    K_QTILE = 64
    K_TILE = 256
    K_DTILE = 512
    N_FULL = 256

    def meta_data():
        dtype = pto.float16
        acc_dtype = pto.float32
        ptr_type = pto.PtrType(dtype)
        i32 = pto.int32
        tv_2d = pto.TensorType(rank=2, dtype=dtype)

        # TODO: omit shape for `SubTensorType`, can merge into one type https://github.com/zhangstevenunity/PTOAS/issues/31
        tile_view_a = pto.SubTensorType(shape=[M_TILE, K_DTILE], dtype=dtype)
        tile_view_b_256 = pto.SubTensorType(shape=[K_TILE, N_FULL], dtype=dtype)
        tile_view_c_256 = pto.SubTensorType(shape=[M_TILE, N_FULL], dtype=dtype)

        b_l1_cfg = pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor", s_fractal_size=512)

        tile_buf_a_l1 = pto.TileBufType(shape=[M_TILE, K_DTILE], dtype=dtype, memory_space="MAT")
        tile_buf_b_l1_256 = pto.TileBufType(shape=[K_TILE, N_FULL], dtype=dtype, memory_space="MAT", config=b_l1_cfg)
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
            "tile_buf_c_256": tile_buf_c_256
        }

    @to_ir_module(meta_data=meta_data)
    def matmul_kernel_ABt(
        a_ptr: "ptr_type",
        b_ptr: "ptr_type",
        c_ptr: "ptr_type",
        m_i32: "i32",
        n_i32: "i32",
        k_i32: "i32"
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

            tvA = pto.as_tensor(tv_2d, ptr=a_ptr, shape=[m_total, k_total], strides=[k_total, c1])
            tvB = pto.as_tensor(tv_2d, ptr=b_ptr, shape=[k_total, n_total], strides=[c1, k_total], layout="DN")
            tvC = pto.as_tensor(tv_2d, ptr=c_ptr, shape=[m_total, n_total], strides=[n_total, c1])

            a_l1 = [pto.alloc_tile(tile_buf_a_l1), pto.alloc_tile(tile_buf_a_l1)]
            b_l1 = [pto.alloc_tile(tile_buf_b_l1_256), pto.alloc_tile(tile_buf_b_l1_256)]
            a_l0 = [pto.alloc_tile(tile_buf_a_l0), pto.alloc_tile(tile_buf_a_l0)]
            b_l0 = [pto.alloc_tile(tile_buf_b_l0_256), pto.alloc_tile(tile_buf_b_l0_256)]
            c_l0 = pto.alloc_tile(tile_buf_c_256)

            pto.record_event("MATMUL", "MOV_M2L", event_id=[0, 1])
            pto.record_event("MOV_M2L", "LOAD", event_id=[0, 1, 2, 3])

            for li in pto.range(bid, core_loop, num_blocks):
                m_idx = li // n_loop
                n_idx = li % n_loop
                m_offset = m_idx * c128
                n_offset = n_idx * c256
                cKT = const(K_TILE)
                cKD = const(K_DTILE)
                cNT = const(N_FULL)

                not_first_tile = li != bid
                with pto.if_context(not_first_tile):
                    pto.wait_event("STORE_ACC", "MATMUL", event_id=0)

                sv_a0 = pto.slice_view(
                    tile_view_a,
                    source=tvA,
                    offsets=[m_offset, c0],
                    sizes=[const(M_TILE), cKD],
                )
                pto.wait_event("MOV_M2L", "LOAD", event_id=0)
                pto.load(sv_a0, a_l1[0])
                pto.record_event("LOAD", "MOV_M2L", event_id=0)

                for k_idx in pto.range(c0, k_dtile_num, c1):
                    k_offset = k_idx * cKD

                    def run_loop_k(curr_id, next_id, a_curr, a_next):
                        # NOTE: here declare nested function so we can reuse for double-buffering
                        is_first_k_tile = k_idx == c0

                        for h in range(2):
                            b_evt = 2 + h
                            h_off = const(h * K_TILE)
                            sv_b = pto.slice_view(
                                tile_view_b_256,
                                source=tvB,
                                offsets=[k_offset + h_off, n_offset],
                                sizes=[cKT, cNT],
                            )

                            pto.wait_event("MOV_M2L", "LOAD", event_id=b_evt)
                            pto.load(sv_b, b_l1[h])
                            pto.record_event("LOAD", "MOV_M2L", event_id=b_evt)

                            for quarter in range(4):
                                # NOTE: here is native Python loop, treats as build-time loop unrolling
                                phase = h * 4 + quarter
                                ping = phase & 1
                                a_col = const(phase * K_QTILE)
                                b_row = const(quarter * K_QTILE)

                                pto.wait_event("MATMUL", "MOV_M2L", event_id=ping)
                                if phase == 0:
                                    pto.wait_event("LOAD", "MOV_M2L", event_id=curr_id)

                                tile.extract(a_curr, c0, a_col, a_l0[ping])
                                if phase == 7:
                                    pto.record_event("MOV_M2L", "LOAD", event_id=curr_id)

                                if quarter == 0:
                                    pto.wait_event("LOAD", "MOV_M2L", event_id=b_evt)

                                tile.extract(b_l1[h], b_row, c0, b_l0[ping])
                                pto.record_event("MOV_M2L", "MATMUL", event_id=0)

                                if quarter == 3:
                                    pto.record_event("MOV_M2L", "LOAD", event_id=b_evt)

                                pto.wait_event("MOV_M2L", "MATMUL", event_id=0)
                                if phase == 0:
                                    pto.cond(
                                        is_first_k_tile,
                                        lambda: tile.matmul(a_l0[ping], b_l0[ping], c_l0),
                                        lambda: tile.matmul_acc(c_l0, a_l0[ping], b_l0[ping], c_l0),
                                    )
                                else:
                                    tile.matmul_acc(c_l0, a_l0[ping], b_l0[ping], c_l0)

                                pto.record_event("MATMUL", "MOV_M2L", event_id=ping)

                        with pto.if_context(k_idx + c1 < k_dtile_num):
                            sv_a_next = pto.slice_view(
                                tile_view_a,
                                source=tvA,
                                offsets=[m_offset, k_offset + cKD],
                                sizes=[const(M_TILE), cKD],
                            )
                            pto.wait_event("MOV_M2L", "LOAD", event_id=next_id)
                            pto.load(sv_a_next, a_next)
                            pto.record_event("LOAD", "MOV_M2L", event_id=next_id)

                    is_curr0 = (k_idx % c2) == c0
                    with pto.if_context(is_curr0, has_else=True) as branch:
                        run_loop_k(0, 1, a_l1[0], a_l1[1])
                    with branch.else_context():
                        run_loop_k(1, 0, a_l1[1], a_l1[0])

                sv_c = pto.slice_view(
                    tile_view_c_256,
                    source=tvC,
                    offsets=[m_offset, n_offset],
                    sizes=[const(M_TILE), cNT],
                )
                pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
                pto.store(c_l0, sv_c)

                with pto.if_context(li + num_blocks < core_loop):
                    pto.record_event("STORE_ACC", "MATMUL", event_id=0)

            pto.wait_event("MOV_M2L", "LOAD", event_id=3)
            pto.wait_event("MOV_M2L", "LOAD", event_id=2)
            pto.wait_event("MOV_M2L", "LOAD", event_id=1)
            pto.wait_event("MOV_M2L", "LOAD", event_id=0)
            pto.wait_event("MATMUL", "MOV_M2L", event_id=0)
            pto.wait_event("MATMUL", "MOV_M2L", event_id=1)

    return matmul_kernel_ABt


if __name__ == "__main__":
    print(build())
