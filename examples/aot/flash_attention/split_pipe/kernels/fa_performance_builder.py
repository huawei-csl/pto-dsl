# Flash-Attention kernel builder. Ports the reference
# `fa_performance_kernel.cpp` (a2a3) software-pipelined schedule onto the
# pto-dsl multi-pipe primitives (ptoas >= 0.29).
#
# This file mirrors the reference C++ scheduler:
#
#   constexpr int qkPreloadNum = 2;   // warmup depth
#
#   /* Prologue: cube emits QK[0..QK_PRELOAD-1]; vec consumes them and
#      pushes P[0..QK_PRELOAD-1]. No PV / gu yet. */
#
#   /* Steady state, tile_id 0..N-1:
#        cube: if (t+QK_PRELOAD < N) compute_qk(t+QK_PRELOAD);
#              compute_pv(tile_id);
#        vec:  if (t+QK_PRELOAD < N) compute_p(t+QK_PRELOAD);
#              compute_gu(tile_id);
#      so vec's softmax for the LOOK-AHEAD tile fills the QK consumption
#      slot WHILE the cube is computing the current PV[t]. The cube
#      stops being blocked on a freshly-pushed P (softmax of t+2 has
#      already pushed P[t+2] into the FIFO by the time cube needs it). */
#
#   /* Epilogue: drain the last QK_PRELOAD tiles' PV / gu. */
#
# The new dependency that the reference solves with `l1_exp_max_ififo`:
# softmax(t+QK_PRELOAD) overwrites the running scratch tile `exp_max`
# (the rescale factor needed by gu(t)). With QK_PRELOAD=2 we therefore
# need a 2-deep ring of `exp_max` tiles (`exp_max_a`, `exp_max_b`). We
# implement the ring by unrolling the steady-state loop in pairs of 2
# iterations: even iters use `exp_max_a`, odd iters use `exp_max_b`.
#
# Other state in the softmax (`new_global_max`, `new_global_sum`) does
# NOT need a ring: it is monotonic accumulator state across all tiles
# and is only read at the very end (the divide into o_tile). The fact
# that softmax(t+2) advances it ahead of gu(t) is harmless because gu
# never reads it.
#
# Sub-block (TILE_UP_DOWN) parallelism is preserved on every pipe op.
#
# Hardware-flag accounting (§3.5): 3 unidir pipes × 2 = 6 flags ≪ 16.

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

import math
import os

const = s.const

# ---------------------------------------------------------------------------
# Static shapes (must match run.py constants)
# ---------------------------------------------------------------------------
S0 = 32  # Q rows per block (vec uses S0_HALF rows; matches cpp_ref cube blocking story)
S0_HALF = S0 // 2  # rows per AIV sub-block
HEAD = 128  # attention head dimension — same as cpp_ref/split_pipe/run.py test_flash(..., head=128)
# cpp_ref benchmarks commonly use tile_s1=512; override with FA_S1_TILE for bring-up/debug (256 fits tighter UB).
S1_TILE = int(os.environ.get("FA_S1_TILE", "512"))
# NUM_TILES is overridable via the FA_NUM_TILES env var so the same builder
# can produce kernels for different sequence lengths
# (S1_TOTAL = S1_TILE * NUM_TILES).
# Constraint: (NUM_TILES - QK_PRELOAD) must be even (steady-state pair unroll).
NUM_TILES = int(os.environ.get("FA_NUM_TILES", "16"))

S1_TOTAL = S1_TILE * NUM_TILES

# Total Q rows: default 3072 = 128 * 24 matches cpp_ref benchmark `s0 = 128 * 24`.
Q_ROWS = int(os.environ.get("FA_Q_ROWS", "3072"))
NUM_Q_BLOCKS = Q_ROWS // S0

# QK preload depth — must be >= 1; reference uses 2. The vec pre-softmaxes
# tiles 0..QK_PRELOAD-1, then the steady-state loop interleaves softmax(t+QK_PRELOAD)
# with gu(t), and the epilogue drains the last QK_PRELOAD gu's.
# (NUM_TILES - QK_PRELOAD) must be even — steady state is pair-unrolled to
# ping-pong the exp_max ring (see below).
QK_PRELOAD = 2
assert (
    NUM_TILES - QK_PRELOAD
) % 2 == 0, "Steady-state pair unrolling requires (NUM_TILES - QK_PRELOAD) % 2 == 0"
STEADY_PAIRS = (NUM_TILES - QK_PRELOAD) // 2

# Per-pipe slot sizes (bytes).
SLOT_SIZE_QK = S0 * S1_TILE * 4  # fp32 QK accumulator
SLOT_SIZE_PV = S0 * HEAD * 4  # fp32 PV accumulator
SLOT_SIZE_P = S0 * S1_TILE * 2  # fp16 P matrix sent vec → cube

# `dir_mask = 1/2` always lowers to slot_num = 8 on a3 (design doc §4.4).
SLOT_NUM = 8
# Kept at 1: bumping to 2 overflows VEC UB at S1_TILE=512.
QK_LOCAL_SLOT_NUM = 1
# PV uses lower-level l2g2l_pipe with local_slot_num=1; the legacy
# aic/aiv_initialize_pipe path forces local = SLOT_NUM = 8 (32 KB MAT)
# whereas local=1 here is just 4 KB.
PV_LOCAL_SLOT_NUM = 1

# GM-staged FIFO bytes / fp32 elements per AIC block.
GM_BYTES_PER_BLOCK = (SLOT_SIZE_QK + SLOT_SIZE_PV + SLOT_SIZE_P) * SLOT_NUM
GM_ELEMS_PER_BLOCK = GM_BYTES_PER_BLOCK // 4
GM_QK_OFF_F32 = 0
GM_PV_OFF_F32 = (SLOT_SIZE_QK * SLOT_NUM) // 4
GM_P_OFF_F32 = GM_PV_OFF_F32 + (SLOT_SIZE_PV * SLOT_NUM) // 4

FIFO_BYTES_QK = SLOT_SIZE_QK * QK_LOCAL_SLOT_NUM
FIFO_BYTES_PV = SLOT_SIZE_PV * PV_LOCAL_SLOT_NUM
FIFO_BYTES_P = SLOT_SIZE_P * SLOT_NUM

# --pto-level=level3 requires explicit byte addresses for every reserve/alloc.
# Vector UB is 192 KiB; recv buffers for `tpop` must fit entirely below that limit.
_TILE_FP32_BYTES = S0_HALF * S1_TILE * 4
_TILE_FP16_BYTES = S0_HALF * S1_TILE * 2
_O_TILE_BYTES = S0_HALF * HEAD * 4
_VEC_RED_STRIDE = 64

# Cube MAT / ACC / LEFT — fits in L1 with K/V sharing RIGHT at addr 0 (single tile footprint).
MAT_Q_OFF = 0
MAT_K_OFF = MAT_Q_OFF + S0 * HEAD * 2
MAT_P_RECV_OFF = MAT_K_OFF + HEAD * S1_TILE * 2
MAT_V_OFF = MAT_P_RECV_OFF + S0 * S1_TILE * 2
_MAT_CUBE_TAIL = MAT_V_OFF + S1_TILE * HEAD * 2
_MAT_ALIGN = 65536
MAT_P_FIFO_OFF = (_MAT_CUBE_TAIL + _MAT_ALIGN - 1) // _MAT_ALIGN * _MAT_ALIGN

LEFT_Q_OFF = 0
LEFT_P_OFF = LEFT_Q_OFF + S0 * HEAD * 2

RIGHT_KV_OFF = 0

ACC_QK_OFF = 0
ACC_PV_OFF = ACC_QK_OFF + S0 * S1_TILE * 4

# Vec scratch (FIFOs then softmax stack). PV `tpop` reuses tmp_tile storage — pipeline serializes gu vs softmax per head32 schedule.
VEC_QK_FIFO_OFF = 0
VEC_PV_FIFO_OFF = VEC_QK_FIFO_OFF + FIFO_BYTES_QK
_VEC_POST_FIFO_OFF = VEC_PV_FIFO_OFF + FIFO_BYTES_PV
VEC_QK_RECV_OFF = _VEC_POST_FIFO_OFF
VEC_TMP_OFF = VEC_QK_RECV_OFF + _TILE_FP32_BYTES
# GU consumes PV before softmax touches tmp again (pairs head32 schedule).
VEC_PV_RECV_OFF = VEC_TMP_OFF
VEC_P_FP32_OFF = VEC_TMP_OFF + _TILE_FP32_BYTES
# fp16 P lives in the **upper half** of the fp32 softmax tile — avoid bisheng assigning
# fp32/fp16 Tiles to the identical UB offset (duplicate TASSIGN breaks TCVT on device).
VEC_P_FP16_OFF = VEC_P_FP32_OFF + _TILE_FP16_BYTES
VEC_O_OFF = VEC_P_FP32_OFF + _TILE_FP32_BYTES
VEC_RED_BASE_OFF = VEC_O_OFF + _O_TILE_BYTES
VEC_NEW_GLOBAL_MAX_OFF = VEC_RED_BASE_OFF + 0 * _VEC_RED_STRIDE
VEC_LOCAL_MAX_OFF = VEC_RED_BASE_OFF + 1 * _VEC_RED_STRIDE
VEC_NEW_GLOBAL_SUM_OFF = VEC_RED_BASE_OFF + 2 * _VEC_RED_STRIDE
VEC_LOCAL_SUM_OFF = VEC_RED_BASE_OFF + 3 * _VEC_RED_STRIDE
VEC_EXP_MAX_A_OFF = VEC_RED_BASE_OFF + 4 * _VEC_RED_STRIDE
VEC_EXP_MAX_B_OFF = VEC_RED_BASE_OFF + 5 * _VEC_RED_STRIDE
_VEC_UB_TAIL = VEC_EXP_MAX_B_OFF + _VEC_RED_STRIDE + 128
assert _VEC_UB_TAIL <= 192 * 1024, (
    f"VEC UB overflow tail={_VEC_UB_TAIL}; reduce S1_TILE or S0"
)
ID_QK = 10  # Cube → Vec, dir_mask = 1 (uses lower-level l2g2l)
ID_PV = 20  # Cube → Vec, dir_mask = 1 (legacy)
ID_P = 30  # Vec  → Cube, dir_mask = 2 (legacy)

SPLIT_UP_DOWN = 1


# ---------------------------------------------------------------------------
# Type definitions (identical to multipipe builder)
# ---------------------------------------------------------------------------
def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    ffts_ty = pto.ffts_type
    ptr_fp16 = pto.PtrType(fp16)
    ptr_fp32 = pto.PtrType(fp32)
    i32 = pto.int32

    qkv_tensor_ty = pto.TensorType(rank=2, dtype=fp16)
    o_tensor_ty = pto.TensorType(rank=2, dtype=fp32)

    q_sub_ty = pto.SubTensorType(shape=[S0, HEAD], dtype=fp16)
    kt_sub_ty = pto.SubTensorType(shape=[HEAD, S1_TILE], dtype=fp16)
    v_sub_ty = pto.SubTensorType(shape=[S1_TILE, HEAD], dtype=fp16)
    o_sub_ty = pto.SubTensorType(shape=[S0, HEAD], dtype=fp32)
    o_sub_half_ty = pto.SubTensorType(shape=[S0_HALF, HEAD], dtype=fp32)

    # --- Cube tile types ---
    q_mat_ty = pto.TileBufType(shape=[S0, HEAD], dtype=fp16, memory_space="MAT")
    q_left_ty = pto.TileBufType(shape=[S0, HEAD], dtype=fp16, memory_space="LEFT")
    k_mat_ty = pto.TileBufType(
        shape=[HEAD, S1_TILE],
        dtype=fp16,
        memory_space="MAT",
        config=pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor"),
    )
    k_right_ty = pto.TileBufType(
        shape=[HEAD, S1_TILE], dtype=fp16, memory_space="RIGHT"
    )
    qk_acc_ty = pto.TileBufType(shape=[S0, S1_TILE], dtype=fp32, memory_space="ACC")
    p_recv_ty = pto.TileBufType(
        shape=[S0, S1_TILE],
        dtype=fp16,
        memory_space="MAT",
    )
    p_left_ty = pto.TileBufType(shape=[S0, S1_TILE], dtype=fp16, memory_space="LEFT")
    v_mat_ty = pto.TileBufType(shape=[S1_TILE, HEAD], dtype=fp16, memory_space="MAT")
    v_right_ty = pto.TileBufType(
        shape=[S1_TILE, HEAD], dtype=fp16, memory_space="RIGHT"
    )
    pv_acc_ty = pto.TileBufType(shape=[S0, HEAD], dtype=fp32, memory_space="ACC")

    # --- Vector tile types (HALF-size — split=1 on every pipe op) ---
    qk_vec_ty = pto.TileBufType(
        shape=[S0_HALF, S1_TILE], dtype=fp32, memory_space="VEC"
    )
    p_fp32_ty = pto.TileBufType(
        shape=[S0_HALF, S1_TILE], dtype=fp32, memory_space="VEC"
    )
    p_fp16_ty = pto.TileBufType(
        shape=[S0_HALF, S1_TILE], dtype=fp16, memory_space="VEC"
    )
    pv_vec_ty = pto.TileBufType(shape=[S0_HALF, HEAD], dtype=fp32, memory_space="VEC")
    red_ty = pto.TileBufType(
        shape=[S0_HALF, 1],
        dtype=fp32,
        memory_space="VEC",
        config=pto.TileBufConfig(blayout="ColMajor", slayout="NoneBox"),
    )
    red_row_ty = pto.TileBufType(
        shape=[1, S0_HALF],
        dtype=fp32,
        memory_space="VEC",
    )
    o_vec_ty = pto.TileBufType(shape=[S0_HALF, HEAD], dtype=fp32, memory_space="VEC")

    return locals()


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------
@to_ir_module(meta_data=meta_data, module=True)
def module():

    # ===================================================================
    # Cube kernel — PRELOAD=2 software-pipelined.
    # ===================================================================
    @pto.func(kernel="cube")
    def cube_kernel(
        gm_slot_buffer: "ptr_fp32",
        gm_q: "ptr_fp16",
        gm_k: "ptr_fp16",
        gm_v: "ptr_fp16",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c2 = const(2)
        cS0 = const(S0)
        cHEAD = const(HEAD)
        cS1_TILE = const(S1_TILE)
        cS1_TOTAL = const(S1_TOTAL)
        cNUM_TILES = const(NUM_TILES)
        cNUM_Q_BLOCKS = const(NUM_Q_BLOCKS)

        num_blocks = s.index_cast(pto.get_block_num())
        bid = s.index_cast(pto.get_block_idx())
        floor_div = cNUM_Q_BLOCKS // num_blocks
        extra = cNUM_Q_BLOCKS % num_blocks
        fat_start = bid * (floor_div + c1)
        thin_start = extra * (floor_div + c1) + (bid - extra) * floor_div
        qb_start = s.select(bid < extra, fat_start, thin_start)
        q_blocks_this_core = s.select(bid < extra, floor_div + c1, floor_div)
        qb_end = qb_start + q_blocks_this_core

        gm_blk_offset = bid * const(GM_ELEMS_PER_BLOCK)
        gm_blk = pto.add_ptr(gm_slot_buffer, gm_blk_offset)
        gm_qk = pto.add_ptr(gm_blk, const(GM_QK_OFF_F32))
        gm_pv = pto.add_ptr(gm_blk, const(GM_PV_OFF_F32))
        gm_p = pto.add_ptr(gm_blk, const(GM_P_OFF_F32))

        # ---- Pipe QK_C2V (lower-level init) ----
        qk_c2v_import = pto.import_reserved_buffer(
            name="fa_qk_c2v_fifo", peer_func="@vector_kernel"
        )
        qk_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_QK,
            slot_num=SLOT_NUM,
            local_slot_num=QK_LOCAL_SLOT_NUM,
            gm_addr=gm_qk,
            local_addr=qk_c2v_import,
        )

        # ---- Pipe PV_C2V (lower-level init: PV_LOCAL_SLOT_NUM VEC slots) ----
        pv_c2v_import = pto.import_reserved_buffer(
            name="fa_pv_c2v_fifo", peer_func="@vector_kernel"
        )
        pv_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_PV,
            slot_num=SLOT_NUM,
            local_slot_num=PV_LOCAL_SLOT_NUM,
            gm_addr=gm_pv,
            local_addr=pv_c2v_import,
        )

        # ---- Pipe P_V2C (id = ID_P) ----
        p_v2c_local = pto.reserve_buffer(
            name="fa_p_v2c_fifo",
            size=FIFO_BYTES_P,
            location="MAT",
            auto_alloc=False,
            base=MAT_P_FIFO_OFF,
        )
        pto.aic_initialize_pipe(
            id=ID_P,
            dir_mask=2,
            slot_size=SLOT_SIZE_P,
            gm_slot_buffer=gm_p,
            c2v_consumer_buf=const(0, s.int32),
            v2c_consumer_buf=p_v2c_local,
            nosplit=False,
        )

        right_base = const(RIGHT_KV_OFF, s.int64)
        q_mat = pto.alloc_tile(q_mat_ty, addr=const(MAT_Q_OFF, s.int64))
        q_left = pto.alloc_tile(q_left_ty, addr=const(LEFT_Q_OFF, s.int64))
        k_mat_s = pto.alloc_tile(k_mat_ty, addr=const(MAT_K_OFF, s.int64))
        k_right_s = pto.alloc_tile(k_right_ty, addr=right_base)
        qk_acc_s = pto.alloc_tile(qk_acc_ty, addr=const(ACC_QK_OFF, s.int64))
        p_recv_s = pto.alloc_tile(p_recv_ty, addr=const(MAT_P_RECV_OFF, s.int64))
        p_left_s = pto.alloc_tile(p_left_ty, addr=const(LEFT_P_OFF, s.int64))
        v_mat_s = pto.alloc_tile(v_mat_ty, addr=const(MAT_V_OFF, s.int64))
        v_right_s = pto.alloc_tile(v_right_ty, addr=right_base)
        pv_acc_s = pto.alloc_tile(pv_acc_ty, addr=const(ACC_PV_OFF, s.int64))
        # Aliasing wrappers: keep the per-iteration `[buf]` indexing pattern
        # in the body even though all slots currently point at one alloc.
        k_mat = [k_mat_s, k_mat_s]
        k_right = [k_right_s, k_right_s]
        qk_acc = [qk_acc_s, qk_acc_s]
        p_recv = [p_recv_s, p_recv_s]
        p_left = [p_left_s, p_left_s]
        v_mat = [v_mat_s, v_mat_s]
        v_right = [v_right_s, v_right_s]
        pv_acc = [pv_acc_s, pv_acc_s]

        cQ_ROWS = const(Q_ROWS)
        tv_q = pto.as_tensor(
            qkv_tensor_ty, ptr=gm_q, shape=[cQ_ROWS, cHEAD], strides=[cHEAD, c1]
        )
        tv_k = pto.as_tensor(
            qkv_tensor_ty,
            ptr=gm_k,
            shape=[cHEAD, cS1_TOTAL],
            strides=[c1, cHEAD],
        )
        tv_v = pto.as_tensor(
            qkv_tensor_ty,
            ptr=gm_v,
            shape=[cS1_TOTAL, cHEAD],
            strides=[cHEAD, c1],
        )

        for qb in pto.range(qb_start, qb_end, c1):
            q_row_off = qb * cS0

            q_view = pto.slice_view(
                q_sub_ty, source=tv_q, offsets=[q_row_off, c0], sizes=[cS0, cHEAD]
            )
            pto.load(q_view, q_mat)
            tile.mov(q_mat, q_left)

            # =================== Cube prologue: emit QK[0..QK_PRELOAD-1] ===================
            # Each prologue QK uses its own k_mat / k_right / qk_acc slot
            # so MTE2 load of K[1] overlaps the M of QK[0].
            for k in range(QK_PRELOAD):
                k_off = const(k * S1_TILE)
                kt_view_k = pto.slice_view(
                    kt_sub_ty,
                    source=tv_k,
                    offsets=[c0, k_off],
                    sizes=[cHEAD, cS1_TILE],
                )
                pto.load(kt_view_k, k_mat[k])
                tile.mov(k_mat[k], k_right[k])
                tile.matmul(q_left, k_right[k], qk_acc[k])
                pto.tpush(qk_acc[k], qk_pipe, SPLIT_UP_DOWN)

            # Preload V[0] for the very first PV.
            v_view_0 = pto.slice_view(
                v_sub_ty,
                source=tv_v,
                offsets=[c0, c0],
                sizes=[cS1_TILE, cHEAD],
            )
            pto.load(v_view_0, v_mat[0])

            # =================== Cube steady state ===================
            # Pair-unrolled. Iter t (parity = t%2 → buffer index `b`):
            #   * load K[next_qk = t+QK_PRELOAD] into k_mat[b]
            #     (next_qk parity equals t parity since QK_PRELOAD == 2)
            #   * pop / mov P[t] into p_left[b]; mov V[t] (in v_mat[b]) → v_right[b]
            #   * preload V[t+1] into v_mat[1-b]
            #   * matmul PV[t] into pv_acc[b]; push
            #   * matmul QK[next_qk] into qk_acc[b]; push
            # Pair handler:
            def emit_cube_step(t_idx, b):
                # next_qk = t_idx + QK_PRELOAD (only used when in main range)
                next_qk = t_idx + const(QK_PRELOAD)
                kt_off = next_qk * cS1_TILE
                kt_view = pto.slice_view(
                    kt_sub_ty,
                    source=tv_k,
                    offsets=[c0, kt_off],
                    sizes=[cHEAD, cS1_TILE],
                )
                pto.load(kt_view, k_mat[b])

                p_raw = pto.tpop_from_aiv(p_recv_ty, SPLIT_UP_DOWN, id=ID_P)
                tile.mov(p_raw, p_left[b])
                pto.tfree_from_aiv(SPLIT_UP_DOWN, id=ID_P)
                tile.mov(v_mat[b], v_right[b])

                v_off = (t_idx + c1) * cS1_TILE
                v_view = pto.slice_view(
                    v_sub_ty,
                    source=tv_v,
                    offsets=[v_off, c0],
                    sizes=[cS1_TILE, cHEAD],
                )
                pto.load(v_view, v_mat[1 - b])

                tile.matmul(p_left[b], v_right[b], pv_acc[b])
                pto.tpush(pv_acc[b], pv_pipe, SPLIT_UP_DOWN)

                tile.mov(k_mat[b], k_right[b])
                tile.matmul(q_left, k_right[b], qk_acc[b])
                pto.tpush(qk_acc[b], qk_pipe, SPLIT_UP_DOWN)

            assert (NUM_TILES - QK_PRELOAD) % 2 == 0
            for p in pto.range(c0, const((NUM_TILES - QK_PRELOAD) // 2), c1):
                t_a = p * c2
                emit_cube_step(t_a, 0)
                t_b = p * c2 + c1
                emit_cube_step(t_b, 1)

            # =================== Cube epilogue: drain last QK_PRELOAD PVs ===================
            # Tile_id range: NUM_TILES-QK_PRELOAD .. NUM_TILES-1.
            # NUM_TILES is even and QK_PRELOAD is even, so the first epilogue
            # tile has parity 0. v_mat[0] holds V[NUM_TILES-QK_PRELOAD] thanks
            # to the last steady-state preload (it loaded V[t_b+1] = V[NUM_TILES-QK_PRELOAD]
            # into v_mat[1-1]=v_mat[0]).
            for k in range(QK_PRELOAD):
                b = k % 2
                p_raw = pto.tpop_from_aiv(p_recv_ty, SPLIT_UP_DOWN, id=ID_P)
                tile.mov(p_raw, p_left[b])
                pto.tfree_from_aiv(SPLIT_UP_DOWN, id=ID_P)
                tile.mov(v_mat[b], v_right[b])
                # Preload V[t+1] into the OPPOSITE slot, only if not the
                # very last tile.
                if k < QK_PRELOAD - 1:
                    next_v_idx = NUM_TILES - QK_PRELOAD + k + 1
                    v_off_k = const(next_v_idx * S1_TILE)
                    v_view_k = pto.slice_view(
                        v_sub_ty,
                        source=tv_v,
                        offsets=[v_off_k, c0],
                        sizes=[cS1_TILE, cHEAD],
                    )
                    pto.load(v_view_k, v_mat[1 - b])
                tile.matmul(p_left[b], v_right[b], pv_acc[b])
                pto.tpush(pv_acc[b], pv_pipe, SPLIT_UP_DOWN)

    # ===================================================================
    # Vector kernel — PRELOAD=2 software-pipelined.
    # ===================================================================
    @pto.func(kernel="vector")
    def vector_kernel(
        gm_slot_buffer: "ptr_fp32",
        gm_o: "ptr_fp32",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c2 = const(2)
        cS0 = const(S0)
        cS0_HALF = const(S0_HALF)
        cHEAD = const(HEAD)
        cNUM_TILES = const(NUM_TILES)
        cNUM_Q_BLOCKS = const(NUM_Q_BLOCKS)

        num_blocks = s.index_cast(pto.get_block_num())
        bid = s.index_cast(pto.get_block_idx())
        floor_div = cNUM_Q_BLOCKS // num_blocks
        extra = cNUM_Q_BLOCKS % num_blocks
        fat_start = bid * (floor_div + c1)
        thin_start = extra * (floor_div + c1) + (bid - extra) * floor_div
        qb_start = s.select(bid < extra, fat_start, thin_start)
        q_blocks_this_core = s.select(bid < extra, floor_div + c1, floor_div)
        qb_end = qb_start + q_blocks_this_core

        gm_blk_offset = bid * const(GM_ELEMS_PER_BLOCK)
        gm_blk = pto.add_ptr(gm_slot_buffer, gm_blk_offset)
        gm_qk = pto.add_ptr(gm_blk, const(GM_QK_OFF_F32))
        gm_pv = pto.add_ptr(gm_blk, const(GM_PV_OFF_F32))
        gm_p = pto.add_ptr(gm_blk, const(GM_P_OFF_F32))

        # ---- Pipe QK_C2V ----
        qk_c2v_local = pto.reserve_buffer(
            name="fa_qk_c2v_fifo",
            size=FIFO_BYTES_QK,
            location="VEC",
            auto_alloc=False,
            base=VEC_QK_FIFO_OFF,
        )
        qk_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_QK,
            slot_num=SLOT_NUM,
            local_slot_num=QK_LOCAL_SLOT_NUM,
            gm_addr=gm_qk,
            local_addr=qk_c2v_local,
        )

        # ---- Pipe PV_C2V (lower-level init: PV_LOCAL_SLOT_NUM VEC slots) ----
        pv_c2v_local = pto.reserve_buffer(
            name="fa_pv_c2v_fifo",
            size=FIFO_BYTES_PV,
            location="VEC",
            auto_alloc=False,
            base=VEC_PV_FIFO_OFF,
        )
        pv_pipe = pto.initialize_l2g2l_pipe(
            dir_mask=1,
            slot_size=SLOT_SIZE_PV,
            slot_num=SLOT_NUM,
            local_slot_num=PV_LOCAL_SLOT_NUM,
            gm_addr=gm_pv,
            local_addr=pv_c2v_local,
        )

        # ---- Pipe P_V2C ----
        p_v2c_import = pto.import_reserved_buffer(
            name="fa_p_v2c_fifo", peer_func="@cube_kernel"
        )
        pto.aiv_initialize_pipe(
            id=ID_P,
            dir_mask=2,
            slot_size=SLOT_SIZE_P,
            gm_slot_buffer=gm_p,
            c2v_consumer_buf=const(0, s.int32),
            v2c_consumer_buf=p_v2c_import,
            nosplit=False,
        )

        sb_idx = s.index_cast(pto.get_subblock_idx())
        row_off_sb = sb_idx * cS0_HALF

        tmp_tile = pto.alloc_tile(qk_vec_ty, addr=const(VEC_TMP_OFF, s.int64))
        p_fp32 = pto.alloc_tile(p_fp32_ty, addr=const(VEC_P_FP32_OFF, s.int64))
        p_fp16 = pto.alloc_tile(p_fp16_ty, addr=const(VEC_P_FP16_OFF, s.int64))
        o_tile = pto.alloc_tile(o_vec_ty, addr=const(VEC_O_OFF, s.int64))
        new_global_max = pto.alloc_tile(
            red_ty, addr=const(VEC_NEW_GLOBAL_MAX_OFF, s.int64)
        )
        local_max = pto.alloc_tile(red_ty, addr=const(VEC_LOCAL_MAX_OFF, s.int64))
        new_global_sum = pto.alloc_tile(
            red_ty, addr=const(VEC_NEW_GLOBAL_SUM_OFF, s.int64)
        )
        local_sum = pto.alloc_tile(red_ty, addr=const(VEC_LOCAL_SUM_OFF, s.int64))
        # Ring of QK_PRELOAD exp_max tiles. With QK_PRELOAD=2 we use a/b
        # ping-pong: even-parity tiles use exp_max_a, odd-parity tiles use
        # exp_max_b. softmax(t) writes the exp_max for tile t into the
        # corresponding slot; gu(t) reads it from the same slot. Because
        # softmax(t+QK_PRELOAD) and gu(t) hit the SAME slot (parity matches),
        # the steady-state loop must do gu(t) BEFORE softmax(t+QK_PRELOAD)
        # to avoid clobbering.
        assert QK_PRELOAD == 2, "exp_max ring is hard-coded to 2 tiles"
        exp_max_a = pto.alloc_tile(red_ty, addr=const(VEC_EXP_MAX_A_OFF, s.int64))
        exp_max_b = pto.alloc_tile(red_ty, addr=const(VEC_EXP_MAX_B_OFF, s.int64))

        scale = const(1.0 / math.sqrt(HEAD), s.float32)
        f32_one = const(1.0, s.float32)

        cQ_ROWS = const(Q_ROWS)
        tv_o = pto.as_tensor(
            o_tensor_ty, ptr=gm_o, shape=[cQ_ROWS, cHEAD], strides=[cHEAD, c1]
        )

        # Helper: emit a softmax step writing into `exp_max_slot`.
        # `is_init` is a Python bool: True only for the very first softmax
        # of the whole block (tile 0) to take the init branch.
        def emit_softmax_step(exp_max_slot, is_init):
            qk_recv = pto.tpop(
                qk_vec_ty,
                qk_pipe,
                SPLIT_UP_DOWN,
                addr=const(VEC_QK_RECV_OFF, s.int64),
            )
            tile.muls(qk_recv, scale, qk_recv)
            tile.row_max(qk_recv, tmp_tile, local_max)

            local_max_r = tile.reshape(red_row_ty, local_max)
            new_global_max_r = tile.reshape(red_row_ty, new_global_max)
            exp_max_r = tile.reshape(red_row_ty, exp_max_slot)
            new_global_sum_r = tile.reshape(red_row_ty, new_global_sum)
            local_sum_r = tile.reshape(red_row_ty, local_sum)

            if is_init:
                tile.row_expand_sub(qk_recv, local_max, p_fp32)
                tile.muls(local_max_r, f32_one, new_global_max_r)
                tile.exp(p_fp32, p_fp32)
                tile.row_sum(p_fp32, tmp_tile, new_global_sum)
            else:
                tile.max(local_max_r, new_global_max_r, local_max_r)
                tile.sub(new_global_max_r, local_max_r, exp_max_r)
                tile.muls(local_max_r, f32_one, new_global_max_r)
                tile.row_expand_sub(qk_recv, local_max, p_fp32)
                tile.exp(exp_max_r, exp_max_r)
                tile.exp(p_fp32, p_fp32)
                tile.mul(new_global_sum_r, exp_max_r, new_global_sum_r)
                tile.row_sum(p_fp32, tmp_tile, local_sum)
                tile.add(new_global_sum_r, local_sum_r, new_global_sum_r)

            tile.cvt(p_fp32, p_fp16)
            pto.tpush_to_aic(p_fp16, SPLIT_UP_DOWN, id=ID_P)
            pto.tfree(qk_pipe, SPLIT_UP_DOWN)

        # Helper: emit a gu step reading from `exp_max_slot`.
        # `is_init` is a Python bool: True only for tile 0 (first PV).
        def emit_gu_step(exp_max_slot, is_init):
            pv_recv = pto.tpop(
                pv_vec_ty,
                pv_pipe,
                SPLIT_UP_DOWN,
                addr=const(VEC_PV_RECV_OFF, s.int64),
            )
            if is_init:
                tile.mov(pv_recv, o_tile)
            else:
                tile.row_expand_mul(o_tile, exp_max_slot, o_tile)
                tile.add(o_tile, pv_recv, o_tile)
            pto.tfree(pv_pipe, SPLIT_UP_DOWN)

        for qb in pto.range(qb_start, qb_end, c1):
            o_row_off = qb * cS0

            # =================== Vec prologue: softmax(0..QK_PRELOAD-1) ===================
            # softmax(0): is_init=True (writes exp_max_a, but exp_max_a for tile 0
            # is unused by gu(0) — gu(0) takes the init branch and just movs PV.
            # Still we must compute it correctly; the init branch doesn't touch exp_max.
            emit_softmax_step(exp_max_a, is_init=True)
            # softmax(1): is_init=False (writes exp_max_b)
            emit_softmax_step(exp_max_b, is_init=False)

            # =================== Vec steady state ===================
            # Pair-unrolled: each `p` iteration handles tiles t_a = 2p, t_b = 2p+1.
            # gu(t_a) reads exp_max_a (set by softmax(t_a) earlier);
            # softmax(t_a+2) writes exp_max_a (matches parity).
            # gu(t_b) reads exp_max_b; softmax(t_b+2) writes exp_max_b.
            # CRITICAL: gu BEFORE softmax in same step to avoid clobbering.
            #
            # First pair (p=0, t_a=0, t_b=1) is Python-unrolled so we can
            # take the `is_init=True` branch on gu(0) (which initializes
            # o_tile via mov rather than rescale+add).
            emit_gu_step(exp_max_a, is_init=True)  # tile 0
            emit_softmax_step(exp_max_a, is_init=False)  # tile 2 → exp_max_a
            emit_gu_step(exp_max_b, is_init=False)  # tile 1
            emit_softmax_step(exp_max_b, is_init=False)  # tile 3 → exp_max_b

            # Remaining pairs (p=1..STEADY_PAIRS-1) inside a runtime loop.
            for p in pto.range(c1, const(STEADY_PAIRS), c1):
                emit_gu_step(exp_max_a, is_init=False)
                emit_softmax_step(exp_max_a, is_init=False)
                emit_gu_step(exp_max_b, is_init=False)
                emit_softmax_step(exp_max_b, is_init=False)

            # =================== Vec epilogue: gu(NUM_TILES-QK_PRELOAD..NUM_TILES-1) ===================
            for k in range(QK_PRELOAD):
                slot = exp_max_a if k % 2 == 0 else exp_max_b
                emit_gu_step(slot, is_init=False)

            tile.row_expand_div(o_tile, new_global_sum, o_tile)

            o_row_off_sb = o_row_off + row_off_sb
            o_view = pto.slice_view(
                o_sub_half_ty,
                source=tv_o,
                offsets=[o_row_off_sb, c0],
                sizes=[cS0_HALF, cHEAD],
            )
            pto.store(o_tile, o_view)

    # ===================================================================
    # Entry point
    # ===================================================================
    @pto.func
    def call_both(
        ffts_addr: "ffts_ty",
        gm_slot_buffer: "ptr_fp32",
        gm_q: "ptr_fp16",
        gm_k: "ptr_fp16",
        gm_v: "ptr_fp16",
        gm_o: "ptr_fp32",
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, gm_q, gm_k, gm_v)
        pto.call(vector_kernel, gm_slot_buffer, gm_o)


if __name__ == "__main__":
    print(module)
