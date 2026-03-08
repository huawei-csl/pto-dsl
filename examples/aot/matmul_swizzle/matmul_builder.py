from mlir.dialects import arith
from mlir.ir import IntegerType

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s


M_TILE = 128
K_QTILE = 64
K_TILE = 256
K_DTILE = 512
N_FULL = 256
N_HALF = 128
SWIZZLE_COUNT = 3

A_FULL_BYTES = M_TILE * K_DTILE * 2
A_SUB_BYTES = M_TILE * K_QTILE * 2
B_L1_START = 2 * A_FULL_BYTES


def build():
    def meta_data():
        dtype = pto.float16
        acc_dtype = pto.float32
        ptr_type = pto.PtrType(dtype)
        i32 = pto.int32
        i64 = IntegerType.get_signless(64)

        tv_a = pto.TensorType(rank=2, dtype=dtype)
        tv_b = pto.TensorType(rank=2, dtype=dtype)
        tv_c = pto.TensorType(rank=2, dtype=dtype)

        tile_view_a = pto.SubTensorType(shape=[M_TILE, K_DTILE], dtype=dtype)
        tile_view_b_256 = pto.SubTensorType(shape=[K_TILE, N_FULL], dtype=dtype)
        tile_view_b_128 = pto.SubTensorType(shape=[K_TILE, N_HALF], dtype=dtype)
        tile_view_c_256 = pto.SubTensorType(shape=[M_TILE, N_FULL], dtype=dtype)
        tile_view_c_128 = pto.SubTensorType(shape=[M_TILE, N_HALF], dtype=dtype)

        b_l1_cfg = pto.TileBufConfig(
            blayout="RowMajor",
            slayout="ColMajor",
            s_fractal_size=512,
        )

        tile_buf_a_l1 = pto.TileBufType(shape=[M_TILE, K_DTILE], dtype=dtype, memory_space="MAT")
        tile_buf_b_l1_256 = pto.TileBufType(
            shape=[K_TILE, N_FULL],
            dtype=dtype,
            memory_space="MAT",
            config=b_l1_cfg,
        )
        tile_buf_b_l1_128 = pto.TileBufType(
            shape=[K_TILE, N_HALF],
            dtype=dtype,
            memory_space="MAT",
            config=b_l1_cfg,
        )
        tile_buf_a_l0 = pto.TileBufType(shape=[M_TILE, K_QTILE], dtype=dtype, memory_space="LEFT")
        tile_buf_b_l0_256 = pto.TileBufType(shape=[K_QTILE, N_FULL], dtype=dtype, memory_space="RIGHT")
        tile_buf_b_l0_128 = pto.TileBufType(shape=[K_QTILE, N_HALF], dtype=dtype, memory_space="RIGHT")
        tile_buf_c_256 = pto.TileBufType(shape=[M_TILE, N_FULL], dtype=acc_dtype, memory_space="ACC")
        tile_buf_c_128 = pto.TileBufType(shape=[M_TILE, N_HALF], dtype=acc_dtype, memory_space="ACC")

        return {
            "ptr_type": ptr_type,
            "i32": i32,
            "i64": i64,
            "tv_a": tv_a,
            "tv_b": tv_b,
            "tv_c": tv_c,
            "tile_view_a": tile_view_a,
            "tile_view_b_256": tile_view_b_256,
            "tile_view_b_128": tile_view_b_128,
            "tile_view_c_256": tile_view_c_256,
            "tile_view_c_128": tile_view_c_128,
            "tile_buf_a_l1": tile_buf_a_l1,
            "tile_buf_b_l1_256": tile_buf_b_l1_256,
            "tile_buf_b_l1_128": tile_buf_b_l1_128,
            "tile_buf_a_l0": tile_buf_a_l0,
            "tile_buf_b_l0_256": tile_buf_b_l0_256,
            "tile_buf_b_l0_128": tile_buf_b_l0_128,
            "tile_buf_c_256": tile_buf_c_256,
            "tile_buf_c_128": tile_buf_c_128,
        }

    const = s.const

    @to_ir_module(meta_data=meta_data)
    def matmul_kernel_ABt(
        a_ptr: "ptr_type",
        b_ptr: "ptr_type",
        c_ptr: "ptr_type",
        m_i32: "i32",
        n_i32: "i32",
        k_i32: "i32",
    ) -> None:
        def i64_const(value: int):
            return arith.ConstantOp(i64, value).result

        def emit_compute_variant(
            n_tile: int,
            b_view_type,
            c_view_type,
            b_l1_type,
            b_l0_type,
            c_type,
            m_offset,
            n_offset,
            k_total,
            n_total,
            k_dtile_num,
            li,
            bid,
            num_blocks,
            tvA,
            tvB,
            tvC,
        ):
            c0 = const(0)
            c1 = const(1)
            c2 = const(2)
            c4 = const(4)
            cKQ = const(K_QTILE)
            cKT = const(K_TILE)
            cKD = const(K_DTILE)
            cNT = const(n_tile)

            a_l1 = [
                pto.alloc_tile(tile_buf_a_l1, addr=i64_const(0)),
                pto.alloc_tile(tile_buf_a_l1, addr=i64_const(A_FULL_BYTES)),
            ]
            b_half_bytes = n_tile * K_TILE * 2
            b_sub_bytes = n_tile * K_QTILE * 2
            b_l1 = [
                pto.alloc_tile(b_l1_type, addr=i64_const(B_L1_START)),
                pto.alloc_tile(b_l1_type, addr=i64_const(B_L1_START + b_half_bytes)),
            ]
            a_l0 = [
                pto.alloc_tile(tile_buf_a_l0, addr=i64_const(0)),
                pto.alloc_tile(tile_buf_a_l0, addr=i64_const(A_SUB_BYTES)),
            ]
            b_l0 = [
                pto.alloc_tile(b_l0_type, addr=i64_const(0)),
                pto.alloc_tile(b_l0_type, addr=i64_const(b_sub_bytes)),
            ]
            c_l0 = pto.alloc_tile(c_type, addr=i64_const(0))

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
                is_curr0 = (k_idx % c2) == c0

                def emit_k_group(curr_id, next_id, a_curr, a_next):
                    is_first_k_tile = k_idx == c0

                    for h in range(2):
                        b_evt = 2 + h
                        h_off = const(h * K_TILE)
                        sv_b = pto.slice_view(
                            b_view_type,
                            source=tvB,
                            offsets=[k_offset + h_off, n_offset],
                            sizes=[cKT, cNT],
                        )

                        pto.wait_event("MOV_M2L", "LOAD", event_id=b_evt)
                        pto.load(sv_b, b_l1[h])
                        pto.record_event("LOAD", "MOV_M2L", event_id=b_evt)

                        for quarter in range(4):
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

                with pto.if_context(is_curr0, has_else=True) as branch:
                    emit_k_group(0, 1, a_l1[0], a_l1[1])
                    with branch.else_context():
                        emit_k_group(1, 0, a_l1[1], a_l1[0])

            sv_c = pto.slice_view(
                c_view_type,
                source=tvC,
                offsets=[m_offset, n_offset],
                sizes=[const(M_TILE), cNT],
            )
            pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
            pto.store(c_l0, sv_c)

            with pto.if_context(li + num_blocks < core_loop):
                pto.record_event("STORE_ACC", "MATMUL", event_id=0)

        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            c2 = const(2)
            c3 = const(3)
            c128 = const(M_TILE)
            c256 = const(N_FULL)
            c128n = const(N_HALF)
            c512 = const(K_DTILE)
            cSwizzle = const(SWIZZLE_COUNT)
            cSwizzleM1 = const(SWIZZLE_COUNT - 1)

            m_total = s.index_cast(m_i32)
            n_total = s.index_cast(n_i32)
            k_total = s.index_cast(k_i32)
            num_blocks = s.index_cast(pto.get_block_num())
            bid = s.index_cast(pto.get_block_idx())

            n_loop = (n_total + c256 - c1) // c256
            m_loop = m_total // c128
            core_loop = n_loop * m_loop
            k_dtile_num = k_total // c512

            tvA = pto.as_tensor(tv_a, ptr=a_ptr, shape=[m_total, k_total], strides=[k_total, c1])
            tvB = pto.as_tensor(
                tv_b,
                ptr=b_ptr,
                shape=[k_total, n_total],
                strides=[c1, k_total],
                layout="DN",
            )
            tvC = pto.as_tensor(tv_c, ptr=c_ptr, shape=[m_total, n_total], strides=[n_total, c1])

            pto.record_event("MATMUL", "MOV_M2L", event_id=[0, 1])
            pto.record_event("MOV_M2L", "LOAD", event_id=[0, 1, 2, 3])

            for li in pto.range(bid, core_loop, num_blocks):
                tile_block_loop = (n_loop + cSwizzleM1) // cSwizzle
                tile_block_span = cSwizzle * m_loop
                tile_block_idx = li // tile_block_span
                in_tile_block_idx = li % tile_block_span
                is_last_block = tile_block_idx == (tile_block_loop - c1)
                n_col_tail = n_loop - cSwizzle * tile_block_idx
                n_col = s.select(is_last_block, n_col_tail, cSwizzle)

                m_idx = in_tile_block_idx // n_col
                n_idx = tile_block_idx * cSwizzle + (in_tile_block_idx % n_col)

                odd_block = (tile_block_idx % c2) == c1
                flipped_m_idx = m_loop - m_idx - c1
                m_idx = s.select(odd_block, flipped_m_idx, m_idx)

                m_offset = m_idx * c128
                n_offset = n_idx * c256
                n_tile_size = s.select(n_offset + c256 > n_total, c128n, c256)

                with pto.if_context(n_tile_size == c256, has_else=True) as branch:
                    emit_compute_variant(
                        N_FULL,
                        tile_view_b_256,
                        tile_view_c_256,
                        tile_buf_b_l1_256,
                        tile_buf_b_l0_256,
                        tile_buf_c_256,
                        m_offset,
                        n_offset,
                        k_total,
                        n_total,
                        k_dtile_num,
                        li,
                        bid,
                        num_blocks,
                        tvA,
                        tvB,
                        tvC,
                    )
                    with branch.else_context():
                        emit_compute_variant(
                            N_HALF,
                            tile_view_b_128,
                            tile_view_c_128,
                            tile_buf_b_l1_128,
                            tile_buf_b_l0_128,
                            tile_buf_c_128,
                            m_offset,
                            n_offset,
                            k_total,
                            n_total,
                            k_dtile_num,
                            li,
                            bid,
                            num_blocks,
                            tvA,
                            tvB,
                            tvC,
                        )

            pto.wait_event("MOV_M2L", "LOAD", event_id=3)
            pto.wait_event("MOV_M2L", "LOAD", event_id=2)
            pto.wait_event("MOV_M2L", "LOAD", event_id=1)
            pto.wait_event("MOV_M2L", "LOAD", event_id=0)
            pto.wait_event("MATMUL", "MOV_M2L", event_id=0)
            pto.wait_event("MATMUL", "MOV_M2L", event_id=1)

    return matmul_kernel_ABt


if __name__ == "__main__":
    print(build())
