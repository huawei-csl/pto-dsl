from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

import math
import os

const = s.const

CUBE_S0 = 128
S0_HALF = CUBE_S0 // 2
VEC_ROWS = S0_HALF // 2
HEAD = 128
CUBE_S1 = 128
TILE_S1 = 256
SUBTILES = TILE_S1 // CUBE_S1

SPLIT_UP_DOWN = 1
SLOT_NUM = 8
LOCAL_SLOT_NUM = 1
QK_PRELOAD = int(os.environ.get("FA_DSL_QK_PRELOAD", "3"))
EXP_RING = int(os.environ.get("FA_DSL_EXP_RING", "3"))

SLOT_SIZE_QK = CUBE_S0 * TILE_S1 * 4
SLOT_SIZE_PV = CUBE_S0 * HEAD * 4
SLOT_SIZE_P = CUBE_S0 * TILE_S1 * 2

GM_BYTES_PER_BLOCK = (SLOT_SIZE_QK + SLOT_SIZE_P + SLOT_SIZE_PV) * SLOT_NUM
GM_ELEMS_PER_BLOCK = GM_BYTES_PER_BLOCK // 4
GM_HALF_ELEMS_PER_BLOCK = GM_BYTES_PER_BLOCK // 2
GM_QK_OFF_F32 = 0
GM_P_OFF_F16 = (SLOT_SIZE_QK * SLOT_NUM) // 2
GM_PV_OFF_F32 = ((SLOT_SIZE_QK + SLOT_SIZE_P) * SLOT_NUM) // 4


def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    ffts_ty = pto.ffts_type
    ptr_fp16 = pto.PtrType(fp16)
    ptr_fp32 = pto.PtrType(fp32)
    i64 = pto.int64

    qkv_tensor_ty = pto.TensorType(rank=2, dtype=fp16)
    o_tensor_ty = pto.TensorType(rank=2, dtype=fp32)
    qk_slot_ty = pto.TensorType(shape=[CUBE_S0, TILE_S1], dtype=fp32)
    qk_vec_slot_ty = pto.TensorType(shape=[S0_HALF, TILE_S1], dtype=fp32)
    p_slot_ty = pto.TensorType(shape=[CUBE_S0, TILE_S1], dtype=fp16)
    p_vec_slot_ty = pto.TensorType(shape=[S0_HALF, TILE_S1], dtype=fp16)
    pv_slot_ty = pto.TensorType(shape=[CUBE_S0, HEAD], dtype=fp32)
    pv_vec_slot_ty = pto.TensorType(shape=[S0_HALF, HEAD], dtype=fp32)

    q_sub_ty = pto.SubTensorType(shape=[CUBE_S0, HEAD], dtype=fp16)
    kt_sub_ty = pto.SubTensorType(shape=[HEAD, CUBE_S1], dtype=fp16)
    v_sub_ty = pto.SubTensorType(shape=[CUBE_S1, HEAD], dtype=fp16)
    o_sub_half_ty = pto.SubTensorType(shape=[S0_HALF, HEAD], dtype=fp32)
    o_sub_vec_ty = pto.SubTensorType(shape=[VEC_ROWS, HEAD], dtype=fp32)
    qk_slot_part_ty = pto.SubTensorType(shape=[CUBE_S0, CUBE_S1], dtype=fp32)
    qk_vec_slot_part_ty = pto.SubTensorType(shape=[VEC_ROWS, TILE_S1], dtype=fp32)
    p_slot_part_ty = pto.SubTensorType(shape=[CUBE_S0, CUBE_S1], dtype=fp16)
    p_vec_slot_part_ty = pto.SubTensorType(shape=[VEC_ROWS, TILE_S1], dtype=fp16)
    pv_slot_part_ty = pto.SubTensorType(shape=[CUBE_S0, HEAD], dtype=fp32)
    pv_vec_slot_part_ty = pto.SubTensorType(shape=[VEC_ROWS, HEAD], dtype=fp32)

    q_mat_ty = pto.TileBufType(shape=[CUBE_S0, HEAD], dtype=fp16, memory_space="MAT")
    q_left_ty = pto.TileBufType(shape=[CUBE_S0, HEAD], dtype=fp16, memory_space="LEFT")
    k_mat_ty = pto.TileBufType(
        shape=[HEAD, CUBE_S1],
        dtype=fp16,
        memory_space="MAT",
        config=pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor"),
    )
    k_right_ty = pto.TileBufType(
        shape=[HEAD, CUBE_S1], dtype=fp16, memory_space="RIGHT"
    )
    qk_acc_ty = pto.TileBufType(
        shape=[CUBE_S0, CUBE_S1], dtype=fp32, memory_space="ACC"
    )

    p_recv_ty = pto.TileBufType(
        shape=[CUBE_S0, CUBE_S1], dtype=fp16, memory_space="MAT"
    )
    p_left_ty = pto.TileBufType(
        shape=[CUBE_S0, CUBE_S1], dtype=fp16, memory_space="LEFT"
    )
    v_mat_ty = pto.TileBufType(shape=[CUBE_S1, HEAD], dtype=fp16, memory_space="MAT")
    v_right_ty = pto.TileBufType(
        shape=[CUBE_S1, HEAD], dtype=fp16, memory_space="RIGHT"
    )
    pv_acc_ty = pto.TileBufType(shape=[CUBE_S0, HEAD], dtype=fp32, memory_space="ACC")

    qk_vec_ty = pto.TileBufType(
        shape=[VEC_ROWS, TILE_S1], dtype=fp32, memory_space="VEC"
    )
    p_fp32_ty = pto.TileBufType(
        shape=[VEC_ROWS, TILE_S1], dtype=fp32, memory_space="VEC"
    )
    p_fp16_ty = pto.TileBufType(
        shape=[VEC_ROWS, TILE_S1], dtype=fp16, memory_space="VEC"
    )
    pv_vec_ty = pto.TileBufType(shape=[VEC_ROWS, HEAD], dtype=fp32, memory_space="VEC")
    o_vec_ty = pto.TileBufType(shape=[VEC_ROWS, HEAD], dtype=fp32, memory_space="VEC")
    tri_ty = pto.TileBufType(shape=[VEC_ROWS, TILE_S1], dtype=fp32, memory_space="VEC")
    red_ty = pto.TileBufType(
        shape=[VEC_ROWS, 1],
        dtype=fp32,
        memory_space="VEC",
        config=pto.TileBufConfig(blayout="ColMajor", slayout="NoneBox"),
    )
    red_row_ty = pto.TileBufType(shape=[1, VEC_ROWS], dtype=fp32, memory_space="VEC")

    return locals()


@to_ir_module(meta_data=meta_data, module=True)
def module():
    @pto.func(kernel="cube")
    def cube_kernel(
        gm_slot_buffer: "ptr_fp32",
        gm_slot_buffer_h: "ptr_fp16",
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
        cCUBE_S1 = const(CUBE_S1)
        cGM_BLOCK = const(GM_ELEMS_PER_BLOCK)
        cGM_BLOCK_H = const(GM_HALF_ELEMS_PER_BLOCK)

        bid = s.index_cast(pto.get_block_idx())
        s0 = s.index_cast(s0_i64)
        s1 = s.index_cast(s1_i64)
        num_tiles_s1 = s1 // cTILE
        q_row_off = bid * cS0
        tiles_this_block = num_tiles_s1

        gm_blk = pto.add_ptr(gm_slot_buffer, bid * cGM_BLOCK)
        gm_blk_h = pto.add_ptr(gm_slot_buffer_h, bid * cGM_BLOCK_H)
        gm_qk = pto.add_ptr(gm_blk, const(GM_QK_OFF_F32))
        gm_p = pto.add_ptr(gm_blk_h, const(GM_P_OFF_F16))
        gm_pv = pto.add_ptr(gm_blk, const(GM_PV_OFF_F32))

        qk_slot_desc = pto.as_tensor(
            qk_slot_ty,
            ptr=gm_qk,
            shape=[cS0, cTILE],
            strides=[cTILE, c1],
        )
        qk_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_QK,
            slot_num=SLOT_NUM,
            gm_addr=qk_slot_desc,
            flag_base=0,
        )

        p_slot_desc = pto.as_tensor(
            p_slot_ty,
            ptr=gm_p,
            shape=[cS0, cTILE],
            strides=[cTILE, c1],
        )
        p_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=2,
            slot_size=SLOT_SIZE_P,
            slot_num=SLOT_NUM,
            gm_addr=p_slot_desc,
            flag_base=2,
        )

        pv_slot_desc = pto.as_tensor(
            pv_slot_ty,
            ptr=gm_pv,
            shape=[cS0, cHEAD],
            strides=[cHEAD, c1],
        )
        pv_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_PV,
            slot_num=SLOT_NUM,
            gm_addr=pv_slot_desc,
            flag_base=4,
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
        p_recv = pto.alloc_tile(p_recv_ty)
        p_left = pto.alloc_tile(p_left_ty)
        v_mat = pto.alloc_tile(v_mat_ty)
        v_right = pto.alloc_tile(v_right_ty)
        pv_acc = pto.alloc_tile(pv_acc_ty)

        q_view = pto.slice_view(
            q_sub_ty, source=tv_q, offsets=[q_row_off, c0], sizes=[cS0, cHEAD]
        )
        pto.load(q_view, q_mat)
        tile.mov(q_mat, q_left)

        qk_entry = pto.declare_global(qk_slot_ty)
        p_entry = pto.declare_global(p_slot_ty)
        pv_entry = pto.declare_global(pv_slot_ty)

        def compute_qk_sub(tile_id, sub):
            tile_col_off = tile_id * cTILE

            k_col_off = tile_col_off + const(sub * CUBE_S1)
            kt_view = pto.slice_view(
                kt_sub_ty,
                source=tv_k,
                offsets=[c0, k_col_off],
                sizes=[cHEAD, cCUBE_S1],
            )
            pto.load(kt_view, k_mat)
            tile.mov(k_mat, k_right)
            tile.matmul(q_left, k_right, qk_acc)
            qk_part = pto.slice_view(
                qk_slot_part_ty,
                source=qk_entry,
                offsets=[c0, const(sub * CUBE_S1)],
                sizes=[cS0, cCUBE_S1],
            )
            pto.store(qk_acc, qk_part)

        def compute_qk(tile_id):
            pto.talloc(qk_entry, qk_pipe, SPLIT_UP_DOWN)
            for sub in range(SUBTILES):
                compute_qk_sub(tile_id, sub)
            pto.tpush(qk_entry, qk_pipe, SPLIT_UP_DOWN)

        def compute_pv_sub(tile_id, sub):
            tile_col_off = tile_id * cTILE

            v_col_off = tile_col_off + const(sub * CUBE_S1)
            v_view = pto.slice_view(
                v_sub_ty,
                source=tv_v,
                offsets=[v_col_off, c0],
                sizes=[cCUBE_S1, cHEAD],
            )
            pto.load(v_view, v_mat)
            p_part = pto.slice_view(
                p_slot_part_ty,
                source=p_entry,
                offsets=[c0, const(sub * CUBE_S1)],
                sizes=[cS0, cCUBE_S1],
            )
            pto.load(p_part, p_recv)
            tile.mov(p_recv, p_left)
            tile.mov(v_mat, v_right)
            if sub == 0:
                tile.matmul(p_left, v_right, pv_acc)
            else:
                tile.matmul_acc(pv_acc, p_left, v_right, pv_acc)

        def push_pv():
            pto.tfree(p_pipe, SPLIT_UP_DOWN, entry=p_entry)

            pto.talloc(pv_entry, pv_pipe, SPLIT_UP_DOWN)
            pv_part = pto.slice_view(
                pv_slot_part_ty,
                source=pv_entry,
                offsets=[c0, c0],
                sizes=[cS0, cHEAD],
            )
            pto.store(pv_acc, pv_part)
            pto.tpush(pv_entry, pv_pipe, SPLIT_UP_DOWN)

        def compute_pv(tile_id):
            pto.tpop_into(p_entry, p_pipe, SPLIT_UP_DOWN)
            for sub in range(SUBTILES):
                compute_pv_sub(tile_id, sub)
            push_pv()

        def compute_qk_pv_interleaved(next_tile, tile_id):
            pto.tpop_into(p_entry, p_pipe, SPLIT_UP_DOWN)
            for sub in range(SUBTILES):
                compute_pv_sub(tile_id, sub)
                if sub == 0:
                    pto.talloc(qk_entry, qk_pipe, SPLIT_UP_DOWN)
                if sub == SUBTILES - 1:
                    push_pv()
                compute_qk_sub(next_tile, sub)
                if sub == SUBTILES - 1:
                    pto.tpush(qk_entry, qk_pipe, SPLIT_UP_DOWN)

        for preload in range(QK_PRELOAD):
            compute_qk(const(preload))

        cPRELOAD = const(QK_PRELOAD)
        steady_end = tiles_this_block - cPRELOAD
        for tile_id in pto.range(c0, steady_end, c1):
            next_tile = tile_id + cPRELOAD
            compute_qk_pv_interleaved(next_tile, tile_id)

        for drain in range(QK_PRELOAD):
            tile_id = steady_end + const(drain)
            compute_pv(tile_id)

    @pto.func(kernel="vector")
    def vector_kernel(
        gm_slot_buffer: "ptr_fp32",
        gm_slot_buffer_h: "ptr_fp16",
        gm_o: "ptr_fp32",
        s0_i64: "i64",
        s1_i64: "i64",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cS0 = const(CUBE_S0)
        cS0_HALF = const(S0_HALF)
        cVEC_ROWS = const(VEC_ROWS)
        cHEAD = const(HEAD)
        cTILE = const(TILE_S1)
        cCUBE_S1 = const(CUBE_S1)
        cGM_BLOCK = const(GM_ELEMS_PER_BLOCK)
        cGM_BLOCK_H = const(GM_HALF_ELEMS_PER_BLOCK)

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
        gm_blk_h = pto.add_ptr(gm_slot_buffer_h, bid * cGM_BLOCK_H)
        gm_qk = pto.add_ptr(gm_blk, const(GM_QK_OFF_F32))
        gm_p = pto.add_ptr(gm_blk_h, const(GM_P_OFF_F16))
        gm_pv = pto.add_ptr(gm_blk, const(GM_PV_OFF_F32))

        qk_slot_desc = pto.as_tensor(
            qk_vec_slot_ty,
            ptr=gm_qk,
            shape=[cS0_HALF, cTILE],
            strides=[cTILE, c1],
        )
        qk_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_QK,
            slot_num=SLOT_NUM,
            gm_addr=qk_slot_desc,
            flag_base=0,
        )

        p_slot_desc = pto.as_tensor(
            p_vec_slot_ty,
            ptr=gm_p,
            shape=[cS0_HALF, cTILE],
            strides=[cTILE, c1],
        )
        p_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=2,
            slot_size=SLOT_SIZE_P,
            slot_num=SLOT_NUM,
            gm_addr=p_slot_desc,
            flag_base=2,
        )

        pv_slot_desc = pto.as_tensor(
            pv_vec_slot_ty,
            ptr=gm_pv,
            shape=[cS0_HALF, cHEAD],
            strides=[cHEAD, c1],
        )
        pv_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_PV,
            slot_num=SLOT_NUM,
            gm_addr=pv_slot_desc,
            flag_base=4,
        )

        tv_o = pto.as_tensor(
            o_tensor_ty, ptr=gm_o, shape=[s0, cHEAD], strides=[cHEAD, c1]
        )

        qk_first = pto.alloc_tile(qk_vec_ty)
        qk_second = pto.alloc_tile(qk_vec_ty)
        reduce_tmp = pto.alloc_tile(p_fp32_ty)
        p_fp16 = pto.alloc_tile(p_fp16_ty)
        o_first = pto.alloc_tile(o_vec_ty)
        o_second = pto.alloc_tile(o_vec_ty)
        recv_first = pto.alloc_tile(pv_vec_ty)
        recv_second = pto.alloc_tile(pv_vec_ty)
        global_max_first = pto.alloc_tile(red_ty)
        global_max_second = pto.alloc_tile(red_ty)
        local_max = pto.alloc_tile(red_ty)
        global_sum_first = pto.alloc_tile(red_ty)
        global_sum_second = pto.alloc_tile(red_ty)
        local_sum = pto.alloc_tile(red_ty)
        exp_max_first_tiles = [pto.alloc_tile(red_ty) for _ in range(EXP_RING)]
        exp_max_second_tiles = [pto.alloc_tile(red_ty) for _ in range(EXP_RING)]

        scale = const(1.0 / math.sqrt(HEAD), s.float32)

        def init_softmax_slice(qk, global_max, global_sum):
            tile.muls(qk, scale, qk)
            tile.row_max(qk, reduce_tmp, global_max)
            tile.row_expand_sub(qk, global_max, qk)
            tile.exp(qk, qk)
            tile.row_sum(qk, reduce_tmp, global_sum)

        def update_softmax_slice(qk, exp_max, global_max, global_sum):
            tile.muls(qk, scale, qk)
            tile.row_max(qk, reduce_tmp, local_max)
            local_max_r = tile.reshape(red_row_ty, local_max)
            exp_max_r = tile.reshape(red_row_ty, exp_max)
            global_max_r = tile.reshape(red_row_ty, global_max)
            global_sum_r = tile.reshape(red_row_ty, global_sum)
            local_sum_r = tile.reshape(red_row_ty, local_sum)

            tile.max(local_max_r, global_max_r, local_max_r)
            tile.sub(global_max_r, local_max_r, exp_max_r)
            tile.exp(exp_max_r, exp_max_r)
            tile.mov(local_max_r, global_max_r)
            tile.mul(global_sum_r, exp_max_r, global_sum_r)

            tile.row_expand_sub(qk, local_max, qk)
            tile.exp(qk, qk)
            tile.row_sum(qk, reduce_tmp, local_sum)
            tile.add(global_sum_r, local_sum_r, global_sum_r)

        def init_softmax(qk0, qk1):
            init_softmax_slice(qk0, global_max_first, global_sum_first)
            init_softmax_slice(qk1, global_max_second, global_sum_second)

        def update_softmax(qk0, qk1, exp_max_first, exp_max_second):
            update_softmax_slice(qk0, exp_max_first, global_max_first, global_sum_first)
            update_softmax_slice(
                qk1, exp_max_second, global_max_second, global_sum_second
            )

        cEXP_RING = const(EXP_RING)

        def dispatch_exp(tile_id, builder, idx=0):
            if idx == EXP_RING - 1:
                builder(exp_max_first_tiles[idx], exp_max_second_tiles[idx])
                return
            with pto.if_context(
                (tile_id % cEXP_RING) == const(idx), has_else=True
            ) as branch:
                builder(exp_max_first_tiles[idx], exp_max_second_tiles[idx])
            with branch.else_context():
                dispatch_exp(tile_id, builder, idx + 1)

        qk_entry = pto.declare_global(qk_vec_slot_ty)
        p_entry = pto.declare_global(p_vec_slot_ty)
        pv_entry = pto.declare_global(pv_vec_slot_ty)

        def pop_qk_slot():
            pto.tpop_into(qk_entry, qk_pipe, SPLIT_UP_DOWN)
            qk_part0 = pto.slice_view(
                qk_vec_slot_part_ty,
                source=qk_entry,
                offsets=[c0, c0],
                sizes=[cVEC_ROWS, cTILE],
            )
            qk_part1 = pto.slice_view(
                qk_vec_slot_part_ty,
                source=qk_entry,
                offsets=[cVEC_ROWS, c0],
                sizes=[cVEC_ROWS, cTILE],
            )
            pto.load(qk_part0, qk_first)
            pto.load(qk_part1, qk_second)
            pto.tfree(qk_pipe, SPLIT_UP_DOWN, entry=qk_entry)

        def push_p_slot():
            pto.talloc(p_entry, p_pipe, SPLIT_UP_DOWN)
            tile.cvt(qk_first, p_fp16, rmode="cast_rint")
            p_part0 = pto.slice_view(
                p_vec_slot_part_ty,
                source=p_entry,
                offsets=[c0, c0],
                sizes=[cVEC_ROWS, cTILE],
            )
            pto.store(p_fp16, p_part0)
            tile.cvt(qk_second, p_fp16, rmode="cast_rint")
            p_part1 = pto.slice_view(
                p_vec_slot_part_ty,
                source=p_entry,
                offsets=[cVEC_ROWS, c0],
                sizes=[cVEC_ROWS, cTILE],
            )
            pto.store(p_fp16, p_part1)
            pto.tpush(p_entry, p_pipe, SPLIT_UP_DOWN)

        def compute_p_init():
            pop_qk_slot()
            init_softmax(qk_first, qk_second)
            push_p_slot()

        def compute_p_update(tile_id, exp_max_first, exp_max_second):
            pop_qk_slot()
            update_softmax(qk_first, qk_second, exp_max_first, exp_max_second)
            push_p_slot()

        def compute_p_update_dispatch(tile_id):
            pop_qk_slot()
            dispatch_exp(
                tile_id,
                lambda exp_max_first, exp_max_second: update_softmax(
                    qk_first,
                    qk_second,
                    exp_max_first,
                    exp_max_second,
                ),
            )
            push_p_slot()

        def pop_pv_slot():
            pto.tpop_into(pv_entry, pv_pipe, SPLIT_UP_DOWN)
            pv_part0 = pto.slice_view(
                pv_vec_slot_part_ty,
                source=pv_entry,
                offsets=[c0, c0],
                sizes=[cVEC_ROWS, cHEAD],
            )
            pv_part1 = pto.slice_view(
                pv_vec_slot_part_ty,
                source=pv_entry,
                offsets=[cVEC_ROWS, c0],
                sizes=[cVEC_ROWS, cHEAD],
            )
            pto.load(pv_part0, recv_first)
            pto.load(pv_part1, recv_second)

        def free_pv_slot():
            pto.tfree(pv_pipe, SPLIT_UP_DOWN, entry=pv_entry)

        def compute_gu_init():
            pop_pv_slot()
            tile.mov(recv_first, o_first)
            tile.mov(recv_second, o_second)
            free_pv_slot()

        def compute_gu_update(exp_max_first, exp_max_second):
            pop_pv_slot()
            tile.row_expand_mul(o_first, exp_max_first, o_first)
            tile.add(o_first, recv_first, o_first)
            tile.row_expand_mul(o_second, exp_max_second, o_second)
            tile.add(o_second, recv_second, o_second)
            free_pv_slot()

        def compute_gu_update_dispatch(tile_id):
            pop_pv_slot()

            def update_o(exp_max_first, exp_max_second):
                tile.row_expand_mul(o_first, exp_max_first, o_first)
                tile.add(o_first, recv_first, o_first)
                tile.row_expand_mul(o_second, exp_max_second, o_second)
                tile.add(o_second, recv_second, o_second)

            dispatch_exp(tile_id, update_o)
            free_pv_slot()

        def compute_gu(tile_id):
            pop_pv_slot()
            with pto.if_context(tile_id == c0, has_else=True) as branch:
                tile.mov(recv_first, o_first)
                tile.mov(recv_second, o_second)
            with branch.else_context():

                def update_o(exp_max_first, exp_max_second):
                    tile.row_expand_mul(o_first, exp_max_first, o_first)
                    tile.add(o_first, recv_first, o_first)
                    tile.row_expand_mul(o_second, exp_max_second, o_second)
                    tile.add(o_second, recv_second, o_second)

                dispatch_exp(tile_id, update_o)
            free_pv_slot()

        compute_p_init()
        for preload in range(1, QK_PRELOAD):
            compute_p_update(
                const(preload),
                exp_max_first_tiles[preload % EXP_RING],
                exp_max_second_tiles[preload % EXP_RING],
            )

        cPRELOAD = const(QK_PRELOAD)
        steady_end = tiles_this_block - cPRELOAD
        with pto.if_context(steady_end > c0):
            compute_gu_init()
            compute_p_update(
                cPRELOAD,
                exp_max_first_tiles[QK_PRELOAD % EXP_RING],
                exp_max_second_tiles[QK_PRELOAD % EXP_RING],
            )

        for tile_id in pto.range(c1, steady_end, c1):
            next_tile = tile_id + cPRELOAD
            compute_gu_update_dispatch(tile_id)
            compute_p_update_dispatch(next_tile)

        for drain in range(QK_PRELOAD):
            tile_id = steady_end + const(drain)
            compute_gu(tile_id)

        tile.row_expand_div(o_first, global_sum_first, o_first)
        tile.row_expand_div(o_second, global_sum_second, o_second)
        o_view0 = pto.slice_view(
            o_sub_vec_ty,
            source=tv_o,
            offsets=[q_row_off_sb, c0],
            sizes=[cVEC_ROWS, cHEAD],
        )
        pto.store(o_first, o_view0)
        o_view1 = pto.slice_view(
            o_sub_vec_ty,
            source=tv_o,
            offsets=[q_row_off_sb + cVEC_ROWS, c0],
            sizes=[cVEC_ROWS, cHEAD],
        )
        pto.store(o_second, o_view1)

    @pto.func
    def call_both(
        ffts_addr: "ffts_ty",
        gm_slot_buffer: "ptr_fp32",
        gm_slot_buffer_h: "ptr_fp16",
        gm_q: "ptr_fp16",
        gm_k: "ptr_fp16",
        gm_v: "ptr_fp16",
        gm_o: "ptr_fp32",
        s0_i64: "i64",
        s1_i64: "i64",
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(
            cube_kernel,
            gm_slot_buffer,
            gm_slot_buffer_h,
            gm_q,
            gm_k,
            gm_v,
            s0_i64,
            s1_i64,
        )
        pto.call(
            vector_kernel,
            gm_slot_buffer,
            gm_slot_buffer_h,
            gm_o,
            s0_i64,
            s1_i64,
        )


if __name__ == "__main__":
    print(module.operation.get_asm(print_generic_op_form=True))
