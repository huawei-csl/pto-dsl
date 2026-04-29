from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

import math

const = s.const

# C++ reference shape, with CUBE_S0/TILE_S1 simplified so the vector side fits
# in UB while preserving the same QK -> softmax/P -> PV -> update dataflow.
CUBE_S0 = 128
S0_HALF = CUBE_S0 // 2
HEAD = 128
TILE_S1 = 128

SPLIT_UP_DOWN = 1
SLOT_NUM = 8
LOCAL_SLOT_NUM = 1

SLOT_SIZE_QK = CUBE_S0 * TILE_S1 * 4
SLOT_SIZE_PV = CUBE_S0 * HEAD * 4
SLOT_SIZE_P = CUBE_S0 * TILE_S1 * 2

GM_BYTES_PER_BLOCK = (SLOT_SIZE_QK + SLOT_SIZE_P) * SLOT_NUM
GM_ELEMS_PER_BLOCK = GM_BYTES_PER_BLOCK // 4
GM_QK_OFF_F32 = 0
GM_P_OFF_F32 = (SLOT_SIZE_QK * SLOT_NUM) // 4

FIFO_BYTES_QK = SLOT_SIZE_QK * LOCAL_SLOT_NUM
FIFO_BYTES_P = SLOT_SIZE_P * LOCAL_SLOT_NUM


def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    ffts_ty = pto.ffts_type
    ptr_fp16 = pto.PtrType(fp16)
    ptr_fp32 = pto.PtrType(fp32)
    i64 = pto.int64

    qkv_tensor_ty = pto.TensorType(rank=2, dtype=fp16)
    o_tensor_ty = pto.TensorType(rank=2, dtype=fp32)

    q_sub_ty = pto.SubTensorType(shape=[CUBE_S0, HEAD], dtype=fp16)
    kt_sub_ty = pto.SubTensorType(shape=[HEAD, TILE_S1], dtype=fp16)
    v_sub_ty = pto.SubTensorType(shape=[TILE_S1, HEAD], dtype=fp16)
    o_sub_half_ty = pto.SubTensorType(shape=[S0_HALF, HEAD], dtype=fp32)

    q_mat_ty = pto.TileBufType(shape=[CUBE_S0, HEAD], dtype=fp16, memory_space="MAT")
    q_left_ty = pto.TileBufType(shape=[CUBE_S0, HEAD], dtype=fp16, memory_space="LEFT")
    k_mat_ty = pto.TileBufType(
        shape=[HEAD, TILE_S1],
        dtype=fp16,
        memory_space="MAT",
        config=pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor"),
    )
    k_right_ty = pto.TileBufType(
        shape=[HEAD, TILE_S1], dtype=fp16, memory_space="RIGHT"
    )
    qk_acc_ty = pto.TileBufType(
        shape=[CUBE_S0, TILE_S1], dtype=fp32, memory_space="ACC"
    )

    p_recv_ty = pto.TileBufType(
        shape=[CUBE_S0, TILE_S1], dtype=fp16, memory_space="MAT"
    )
    p_left_ty = pto.TileBufType(
        shape=[CUBE_S0, TILE_S1], dtype=fp16, memory_space="LEFT"
    )
    v_mat_ty = pto.TileBufType(shape=[TILE_S1, HEAD], dtype=fp16, memory_space="MAT")
    v_right_ty = pto.TileBufType(
        shape=[TILE_S1, HEAD], dtype=fp16, memory_space="RIGHT"
    )
    pv_acc_ty = pto.TileBufType(shape=[CUBE_S0, HEAD], dtype=fp32, memory_space="ACC")

    qk_vec_ty = pto.TileBufType(
        shape=[S0_HALF, TILE_S1], dtype=fp32, memory_space="VEC"
    )
    p_fp32_ty = pto.TileBufType(
        shape=[S0_HALF, TILE_S1], dtype=fp32, memory_space="VEC"
    )
    p_fp16_ty = pto.TileBufType(
        shape=[S0_HALF, TILE_S1], dtype=fp16, memory_space="VEC"
    )
    pv_vec_ty = pto.TileBufType(shape=[S0_HALF, HEAD], dtype=fp32, memory_space="VEC")
    o_vec_ty = pto.TileBufType(shape=[S0_HALF, HEAD], dtype=fp32, memory_space="VEC")
    tri_ty = pto.TileBufType(shape=[S0_HALF, TILE_S1], dtype=fp32, memory_space="VEC")
    red_ty = pto.TileBufType(
        shape=[S0_HALF, 1],
        dtype=fp32,
        memory_space="VEC",
        config=pto.TileBufConfig(blayout="ColMajor", slayout="NoneBox"),
    )
    red_row_ty = pto.TileBufType(shape=[1, S0_HALF], dtype=fp32, memory_space="VEC")

    return locals()


@to_ir_module(meta_data=meta_data, module=True)
def module():
    @pto.func(kernel="cube")
    def cube_kernel(
        gm_slot_buffer: "ptr_fp32",
        gm_q: "ptr_fp16",
        gm_k: "ptr_fp16",
        gm_v: "ptr_fp16",
        s0_i64: "i64",
        s1_i64: "i64",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cS0 = const(CUBE_S0)
        cHEAD = const(HEAD)
        cTILE = const(TILE_S1)
        cGM_BLOCK = const(GM_ELEMS_PER_BLOCK)

        bid = s.index_cast(pto.get_block_idx())
        s0 = s.index_cast(s0_i64)
        s1 = s.index_cast(s1_i64)
        num_tiles_s1 = s1 // cTILE
        q_row_off = bid * cS0
        tiles_this_block = num_tiles_s1

        gm_blk = pto.add_ptr(gm_slot_buffer, bid * cGM_BLOCK)
        gm_qk = pto.add_ptr(gm_blk, const(GM_QK_OFF_F32))
        gm_p = pto.add_ptr(gm_blk, const(GM_P_OFF_F32))

        qk_c2v_import = pto.import_reserved_buffer(
            name="fa_dsl_c2v_fifo", peer_func="@vector_kernel"
        )
        qk_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_QK,
            slot_num=SLOT_NUM,
            local_slot_num=LOCAL_SLOT_NUM,
            gm_addr=gm_qk,
            local_addr=qk_c2v_import,
        )

        p_v2c_local = pto.reserve_buffer(
            name="fa_dsl_p_v2c_fifo", size=FIFO_BYTES_P, location="MAT"
        )
        p_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=2,
            slot_size=SLOT_SIZE_P,
            slot_num=SLOT_NUM,
            local_slot_num=LOCAL_SLOT_NUM,
            gm_addr=gm_p,
            local_addr=p_v2c_local,
        )

        tv_q = pto.as_tensor(
            qkv_tensor_ty, ptr=gm_q, shape=[s0, cHEAD], strides=[cHEAD, c1]
        )
        tv_k = pto.as_tensor(
            qkv_tensor_ty,
            ptr=gm_k,
            shape=[cHEAD, s1],
            strides=[c1, cHEAD],
            layout="DN",
        )
        tv_v = pto.as_tensor(
            qkv_tensor_ty, ptr=gm_v, shape=[s1, cHEAD], strides=[cHEAD, c1]
        )

        q_mat = pto.alloc_tile(q_mat_ty)
        q_left = pto.alloc_tile(q_left_ty)
        k_mat = pto.alloc_tile(k_mat_ty)
        k_right = pto.alloc_tile(k_right_ty)
        qk_acc = pto.alloc_tile(qk_acc_ty)
        p_left = pto.alloc_tile(p_left_ty)
        v_mat = pto.alloc_tile(v_mat_ty)
        v_right = pto.alloc_tile(v_right_ty)
        pv_acc = pto.alloc_tile(pv_acc_ty)

        q_view = pto.slice_view(
            q_sub_ty, source=tv_q, offsets=[q_row_off, c0], sizes=[cS0, cHEAD]
        )
        pto.load(q_view, q_mat)
        tile.mov(q_mat, q_left)

        for tile_id in pto.range(c0, tiles_this_block, c1):
            k_col_off = tile_id * cTILE
            kt_view = pto.slice_view(
                kt_sub_ty, source=tv_k, offsets=[c0, k_col_off], sizes=[cHEAD, cTILE]
            )
            pto.load(kt_view, k_mat)
            tile.mov(k_mat, k_right)
            tile.matmul(q_left, k_right, qk_acc)
            pto.tpush(qk_acc, qk_pipe, SPLIT_UP_DOWN)

            p_recv = pto.tpop(p_recv_ty, p_pipe, SPLIT_UP_DOWN)
            tile.mov(p_recv, p_left)
            pto.tfree(p_pipe, SPLIT_UP_DOWN)

            v_view = pto.slice_view(
                v_sub_ty, source=tv_v, offsets=[k_col_off, c0], sizes=[cTILE, cHEAD]
            )
            pto.load(v_view, v_mat)
            tile.mov(v_mat, v_right)
            tile.matmul(p_left, v_right, pv_acc)
            pto.tpush(pv_acc, qk_pipe, SPLIT_UP_DOWN)

    @pto.func(kernel="vector")
    def vector_kernel(
        gm_slot_buffer: "ptr_fp32",
        gm_o: "ptr_fp32",
        s0_i64: "i64",
        s1_i64: "i64",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cS0 = const(CUBE_S0)
        cS0_HALF = const(S0_HALF)
        cHEAD = const(HEAD)
        cTILE = const(TILE_S1)
        cGM_BLOCK = const(GM_ELEMS_PER_BLOCK)

        bid = s.index_cast(pto.get_block_idx())
        sbid = s.index_cast(pto.get_subblock_idx())
        s0 = s.index_cast(s0_i64)
        s1 = s.index_cast(s1_i64)
        num_tiles_s1 = s1 // cTILE
        q_row_off = bid * cS0
        row_off_sb = sbid * cS0_HALF
        q_row_off_sb = q_row_off + row_off_sb
        tiles_this_block = num_tiles_s1

        gm_blk = pto.add_ptr(gm_slot_buffer, bid * cGM_BLOCK)
        gm_qk = pto.add_ptr(gm_blk, const(GM_QK_OFF_F32))
        gm_p = pto.add_ptr(gm_blk, const(GM_P_OFF_F32))

        qk_c2v_local = pto.reserve_buffer(
            name="fa_dsl_c2v_fifo",
            size=FIFO_BYTES_QK,
            location="VEC",
        )
        qk_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_QK,
            slot_num=SLOT_NUM,
            local_slot_num=LOCAL_SLOT_NUM,
            gm_addr=gm_qk,
            local_addr=qk_c2v_local,
        )

        p_v2c_import = pto.import_reserved_buffer(
            name="fa_dsl_p_v2c_fifo", peer_func="@cube_kernel"
        )
        p_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=2,
            slot_size=SLOT_SIZE_P,
            slot_num=SLOT_NUM,
            local_slot_num=LOCAL_SLOT_NUM,
            gm_addr=gm_p,
            local_addr=p_v2c_import,
        )

        tv_o = pto.as_tensor(
            o_tensor_ty, ptr=gm_o, shape=[s0, cHEAD], strides=[cHEAD, c1]
        )

        p_fp32 = pto.alloc_tile(p_fp32_ty)
        p_fp16 = pto.alloc_tile(p_fp16_ty)
        o_tile = pto.alloc_tile(o_vec_ty)
        recv_tile = pto.alloc_tile(qk_vec_ty)
        global_max = pto.alloc_tile(red_ty)
        local_max = pto.alloc_tile(red_ty)
        global_sum = pto.alloc_tile(red_ty)
        local_sum = pto.alloc_tile(red_ty)
        exp_max = pto.alloc_tile(red_ty)

        scale = const(1.0 / math.sqrt(HEAD), s.float32)
        f32_one = const(1.0, s.float32)

        def init_softmax(qk_tile):
            tile.row_max(qk_tile, p_fp32, global_max)
            tile.row_expand_sub(qk_tile, global_max, p_fp32)
            tile.muls(p_fp32, scale, p_fp32)
            tile.exp(p_fp32, p_fp32)
            tile.row_sum(p_fp32, qk_tile, global_sum)

        def update_softmax(qk_tile):
            tile.row_max(qk_tile, p_fp32, local_max)
            local_max_r = tile.reshape(red_row_ty, local_max)
            global_max_r = tile.reshape(red_row_ty, global_max)
            exp_max_r = tile.reshape(red_row_ty, exp_max)
            global_sum_r = tile.reshape(red_row_ty, global_sum)
            local_sum_r = tile.reshape(red_row_ty, local_sum)

            tile.max(local_max_r, global_max_r, local_max_r)
            tile.sub(global_max_r, local_max_r, exp_max_r)
            tile.muls(exp_max_r, scale, exp_max_r)
            tile.exp(exp_max_r, exp_max_r)
            tile.muls(local_max_r, f32_one, global_max_r)

            tile.row_expand_sub(qk_tile, local_max, p_fp32)
            tile.muls(p_fp32, scale, p_fp32)
            tile.exp(p_fp32, p_fp32)
            tile.mul(global_sum_r, exp_max_r, global_sum_r)
            tile.row_sum(p_fp32, qk_tile, local_sum)
            tile.add(global_sum_r, local_sum_r, global_sum_r)

        for tile_id in pto.range(c0, tiles_this_block, c1):
            pto.tpop_into(recv_tile, qk_pipe, SPLIT_UP_DOWN)

            with pto.if_context(tile_id == c0, has_else=True) as branch:
                init_softmax(recv_tile)
            with branch.else_context():
                update_softmax(recv_tile)

            tile.cvt(p_fp32, p_fp16, rmode="cast_rint")
            pto.tpush(p_fp16, p_pipe, SPLIT_UP_DOWN)
            pto.tfree(qk_pipe, SPLIT_UP_DOWN)

            pto.tpop_into(recv_tile, qk_pipe, SPLIT_UP_DOWN)
            with pto.if_context(tile_id == c0, has_else=True) as branch:
                tile.mov(recv_tile, o_tile)
            with branch.else_context():
                tile.row_expand_mul(o_tile, exp_max, o_tile)
                tile.add(o_tile, recv_tile, o_tile)
            pto.tfree(qk_pipe, SPLIT_UP_DOWN)

        tile.row_expand_div(o_tile, global_sum, o_tile)
        o_view = pto.slice_view(
            o_sub_half_ty,
            source=tv_o,
            offsets=[q_row_off_sb, c0],
            sizes=[cS0_HALF, cHEAD],
        )
        pto.store(o_tile, o_view)

    @pto.func
    def call_both(
        ffts_addr: "ffts_ty",
        gm_slot_buffer: "ptr_fp32",
        gm_q: "ptr_fp16",
        gm_k: "ptr_fp16",
        gm_v: "ptr_fp16",
        gm_o: "ptr_fp32",
        s0_i64: "i64",
        s1_i64: "i64",
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, gm_q, gm_k, gm_v, s0_i64, s1_i64)
        pto.call(vector_kernel, gm_slot_buffer, gm_o, s0_i64, s1_i64)


if __name__ == "__main__":
    print(module)
