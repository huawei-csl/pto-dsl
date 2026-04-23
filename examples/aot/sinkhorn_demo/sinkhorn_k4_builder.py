"""
PTO-DSL builder for the minimal fp16 Sinkhorn K=4 kernel.

Vector section, K=4, UB tiles 16×16 with valid 4×4 math (same iteration
order as the usual hand-tuned C++ demo).  Full rebuild::

    ./compile.sh

The emitted ``__global__`` entry point is ``sinkhorn_k4_fp16``;
``caller_sinkhorn_k4.cpp`` wraps it as ``call_sinkhorn`` for the ctypes host ABI.
"""

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

# ---- Kernel constants ----
K = 4
TILE_DIM = 16


def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    i32 = pto.int32
    ptr_fp16 = pto.PtrType(fp16)
    tensor2_fp16 = pto.TensorType(rank=2, dtype=fp16)
    sub_kk_fp16 = pto.SubTensorType(shape=[K, K], dtype=fp16)

    row_cfg = pto.TileBufConfig()
    col_cfg = pto.TileBufConfig(blayout="ColMajor")

    # Full UB tile (16×16) — row-major, fp16 alignment for vector loads.
    matrix_fp16 = pto.TileBufType(
        shape=[TILE_DIM, TILE_DIM],
        valid_shape=[TILE_DIM, TILE_DIM],
        dtype=fp16,
        memory_space="VEC",
        config=row_cfg,
    )

    # TROWMAX / TROWSUM destination: one scalar per row → [TILE_DIM, 1] ColMajor.
    col_vec_fp16 = pto.TileBufType(
        shape=[TILE_DIM, 1],
        valid_shape=[-1, -1],
        dtype=fp16,
        memory_space="VEC",
        config=col_cfg,
    )

    # TCOLSUM destination: one scalar per column → [1, TILE_DIM] RowMajor.
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
    cTILE = const(TILE_DIM)
    f0 = const(0.0, s.float32)
    f0_h = s.truncf(f0, s.float16)

    nm = s.index_cast(num_matrices_i32)
    repeat_idx = s.index_cast(repeat_i32)
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

        mat_full = pto.alloc_tile(matrix_fp16)
        scratch_full = pto.alloc_tile(matrix_fp16)
        row_stat = pto.alloc_tile(col_vec_fp16, valid_row=cK, valid_col=c1)
        col_stat = pto.alloc_tile(row_vec_fp16, valid_row=c1, valid_col=cK)

        mat_kk = tile.subset(mat_full, [c0, c0], [K, K])
        # Match hand-tuned C++: TADDS on the first K×TILE_DIM rows (eps on padding too).
        mat_eps_rows = tile.subset(mat_full, [c0, c0], [K, TILE_DIM])
        scratch_kk = tile.subset(scratch_full, [c0, c0], [K, K])

        for mi in pto.range(wid, nm, num_workers):
            row0 = mi * cK
            gm_in = pto.slice_view(
                sub_kk_fp16,
                source=tv_in,
                offsets=[row0, c0],
                sizes=[cK, cK],
            )
            gm_out = pto.slice_view(
                sub_kk_fp16,
                source=tv_out,
                offsets=[row0, c0],
                sizes=[cK, cK],
            )

            tile.muls(mat_full, f0_h, mat_full)
            pto.load(gm_in, mat_kk)

            tile.row_max(mat_kk, scratch_kk, row_stat)
            tile.row_expand_sub(mat_kk, row_stat, mat_kk)
            tile.exp(mat_kk, mat_kk)

            tile.row_sum(mat_kk, scratch_kk, row_stat)
            tile.row_expand_div(mat_kk, row_stat, mat_kk)

            tile.adds(mat_eps_rows, eps_h, mat_eps_rows)

            tile.col_sum(mat_kk, scratch_kk, col_stat)
            tile.adds(col_stat, eps_h, col_stat)
            tile.col_expand_div(mat_kk, col_stat, mat_kk)

            for _ in pto.range(c1, repeat_idx, c1):
                tile.row_sum(mat_kk, scratch_kk, row_stat)
                tile.adds(row_stat, eps_h, row_stat)
                tile.row_expand_div(mat_kk, row_stat, mat_kk)

                tile.col_sum(mat_kk, scratch_kk, col_stat)
                tile.adds(col_stat, eps_h, col_stat)
                tile.col_expand_div(mat_kk, col_stat, mat_kk)

            pto.store(mat_kk, gm_out)


if __name__ == "__main__":
    print(sinkhorn_k4_fp16)
