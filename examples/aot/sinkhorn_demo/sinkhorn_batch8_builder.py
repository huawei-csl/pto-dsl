"""
PTO-DSL builder for the fp16 Sinkhorn K=4 kernel with BATCH=8 stacked loads
(same algorithm as ``cpp_ref/kernel_sinkhorn.cpp`` in this demo).

Each vector core processes up to eight 4×4 matrices per group: one load / one
store for the stack, batched row-wise ops (softmax, row-normalize), and a small
loop for per-matrix column-normalize.

The Sinkhorn tail (``repeat - 1`` row/col iterations) is a **static** Python
``range(1, repeat)`` so it unrolls at IR build time; ``repeat`` must match the
host ``repeat_i32`` argument (10 in this demo).
"""

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

K = 4
TILE_DIM = 16
BATCH = 8
STACK_ROWS = BATCH * K  # 32

# Static Sinkhorn outer iteration count for the tail loop (build-time unroll).
# Host code must pass the same value as ``repeat_i32`` (this demo uses 10).
repeat = 10


def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    i32 = pto.int32
    ptr_fp16 = pto.PtrType(fp16)
    tensor2_fp16 = pto.TensorType(rank=2, dtype=fp16)
    sub_stack_rk_fp16 = pto.SubTensorType(shape=[STACK_ROWS, K], dtype=fp16)

    row_cfg = pto.TileBufConfig()
    col_cfg = pto.TileBufConfig(blayout="ColMajor")

    matrix_stack_fp16 = pto.TileBufType(
        shape=[STACK_ROWS, TILE_DIM],
        valid_shape=[-1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=row_cfg,
    )

    col_vec_stack_fp16 = pto.TileBufType(
        shape=[STACK_ROWS, 1],
        valid_shape=[-1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=col_cfg,
    )

    row_vec_fp16 = pto.TileBufType(
        shape=[1, TILE_DIM],
        valid_shape=[-1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=row_cfg,
    )

    return locals()


@to_ir_module(meta_data=meta_data)
def sinkhorn_k4_fp16(
    input_ptr: "ptr_fp16",
    output_ptr: "ptr_fp16",
    num_matrices_i32: "i32",
    repeat_i32: "i32",
    eps: "fp32",
) -> None:
    c0 = const(0)
    c1 = const(1)
    cK = const(K)
    cBATCH = const(BATCH)
    f0 = const(0.0, s.float32)
    f0_h = s.truncf(f0, s.float16)

    nm = s.index_cast(num_matrices_i32)
    eps_h = s.truncf(eps, s.float16)

    with pto.vector_section():
        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        num_blocks = pto.get_block_num()

        wid = s.index_cast(cid * sub_bnum + sub_bid)
        num_workers = s.index_cast(num_blocks * sub_bnum)

        n_rows = nm * cK

        tv_in = pto.as_tensor(
            tensor2_fp16,
            ptr=input_ptr,
            shape=[n_rows, cK],
            strides=[cK, c1],
        )
        tv_out = pto.as_tensor(
            tensor2_fp16,
            ptr=output_ptr,
            shape=[n_rows, cK],
            strides=[cK, c1],
        )

        stride_group = num_workers * cBATCH

        for group in pto.range(wid * cBATCH, nm, stride_group):
            nm_minus_g = nm - group
            actual = s.min_u(cBATCH, nm_minus_g)
            rows = actual * cK

            row0 = group * cK

            gm_in = pto.slice_view(
                sub_stack_rk_fp16,
                source=tv_in,
                offsets=[row0, c0],
                sizes=[rows, cK],
            )
            gm_out = pto.slice_view(
                sub_stack_rk_fp16,
                source=tv_out,
                offsets=[row0, c0],
                sizes=[rows, cK],
            )

            mat_stack = pto.alloc_tile(
                matrix_stack_fp16, valid_row=rows, valid_col=cK
            )
            scratch_stack = pto.alloc_tile(
                matrix_stack_fp16, valid_row=rows, valid_col=cK
            )
            row_stat = pto.alloc_tile(
                col_vec_stack_fp16, valid_row=rows, valid_col=c1
            )
            col_stat = pto.alloc_tile(row_vec_fp16, valid_row=c1, valid_col=cK)

            mat_wide = tile.subview(mat_stack, [c0, c0], [STACK_ROWS, TILE_DIM])
            tile.muls(mat_wide, f0_h, mat_wide)

            mat_rk = tile.subview(mat_stack, [c0, c0], [STACK_ROWS, K])
            pto.load(gm_in, mat_rk)

            tile.row_max(mat_stack, scratch_stack, row_stat)
            tile.row_expand_sub(mat_stack, row_stat, mat_stack)
            tile.exp(mat_stack, mat_stack)
            tile.row_sum(mat_stack, scratch_stack, row_stat)
            tile.row_expand_div(mat_stack, row_stat, mat_stack)

            mat_eps = tile.subview(mat_stack, [c0, c0], [STACK_ROWS, TILE_DIM])
            tile.adds(mat_eps, eps_h, mat_eps)

            for m in pto.range(c0, actual, c1):
                m_row = m * cK
                mat_m = tile.subview(mat_stack, [m_row, c0], [K, K])
                scratch_m = tile.subview(scratch_stack, [m_row, c0], [K, K])
                tile.col_sum(mat_m, scratch_m, col_stat)
                tile.adds(col_stat, eps_h, col_stat)
                tile.col_expand_div(mat_m, col_stat, mat_m)

            for _ in range(1, repeat):
                tile.row_sum(mat_stack, scratch_stack, row_stat)
                tile.adds(row_stat, eps_h, row_stat)
                tile.row_expand_div(mat_stack, row_stat, mat_stack)

                for m in pto.range(c0, actual, c1):
                    m_row = m * cK
                    mat_m = tile.subview(mat_stack, [m_row, c0], [K, K])
                    scratch_m = tile.subview(scratch_stack, [m_row, c0], [K, K])
                    tile.col_sum(mat_m, scratch_m, col_stat)
                    tile.adds(col_stat, eps_h, col_stat)
                    tile.col_expand_div(mat_m, col_stat, mat_m)

            pto.store(mat_rk, gm_out)


if __name__ == "__main__":
    print(sinkhorn_k4_fp16)
