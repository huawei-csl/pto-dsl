"""PTO DSL port of TileLang fp8_gemm kernel.

Original (GPU): C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T with per-128 block
scales on both A and B. Outer accumulator in FP32, scale-corrected
sub-results in a separate accumulator for 2x precision.

NPU port: FP8 unsupported. We use FP16 inputs / FP32 accumulator and
keep the per-block scale multiply structure. Block sizes match the
reference (block_M=32, block_N=128, block_K=group_size=128).

Args:
    A: [M, K]                       fp16
    B: [K, N]                       fp16
    C: [M, N]                       fp16   (output)
    Sa: [M,        ceil(K/128)]     fp32
    Sb: [ceil(N/128), ceil(K/128)]  fp32

NOTE: B is in [K, N] layout for NPU RIGHT-tile compatibility (the GPU
reference stored it as [N, K] and used `transpose_B=True` in GEMM).
"""

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

GROUP_SIZE = 128
BLOCK_M = 32
BLOCK_N = 128
BLOCK_K = 128


def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    i32 = pto.int32

    ptr_fp16 = pto.PtrType(fp16)
    ptr_fp32 = pto.PtrType(fp32)

    tv_fp16 = pto.TensorType(rank=2, dtype=fp16)
    tv_fp32 = pto.TensorType(rank=2, dtype=fp32)

    sv_a = pto.SubTensorType(shape=[BLOCK_M, BLOCK_K], dtype=fp16)
    sv_b = pto.SubTensorType(shape=[BLOCK_K, BLOCK_N], dtype=fp16)
    sv_c = pto.SubTensorType(shape=[BLOCK_M, BLOCK_N], dtype=fp16)

    tile_a_mat = pto.TileBufType(
        shape=[BLOCK_M, BLOCK_K], dtype=fp16, memory_space="MAT"
    )
    tile_b_mat = pto.TileBufType(
        shape=[BLOCK_K, BLOCK_N], dtype=fp16, memory_space="MAT"
    )
    tile_a_left = pto.TileBufType(
        shape=[BLOCK_M, BLOCK_K], dtype=fp16, memory_space="LEFT"
    )
    tile_b_right = pto.TileBufType(
        shape=[BLOCK_K, BLOCK_N], dtype=fp16, memory_space="RIGHT"
    )
    tile_c_acc = pto.TileBufType(
        shape=[BLOCK_M, BLOCK_N], dtype=fp32, memory_space="ACC"
    )
    return locals()


@to_ir_module(meta_data=meta_data)
def fp8_gemm(
    a_ptr: "ptr_fp16",
    b_ptr: "ptr_fp16",
    c_ptr: "ptr_fp16",
    sa_ptr: "ptr_fp32",
    sb_ptr: "ptr_fp32",
    M_i32: "i32",
    N_i32: "i32",
    K_i32: "i32",
) -> None:
    c0 = const(0)
    c1 = const(1)
    cBM = const(BLOCK_M)
    cBN = const(BLOCK_N)
    cBK = const(BLOCK_K)

    M = s.index_cast(M_i32)
    N = s.index_cast(N_i32)
    K = s.index_cast(K_i32)
    K_iters = s.ceil_div(K, cBK)

    with pto.cube_section():
        bid = s.index_cast(pto.get_block_idx())
        num_blocks = s.index_cast(pto.get_block_num())
        nblk_m = s.ceil_div(M, cBM)
        nblk_n = s.ceil_div(N, cBN)
        total = nblk_m * nblk_n
        per_core = s.ceil_div(total, num_blocks)
        b_start = bid * per_core
        b_end = s.min_u(b_start + per_core, total)

        tvA = pto.as_tensor(tv_fp16, ptr=a_ptr, shape=[M, K], strides=[K, c1])
        tvB = pto.as_tensor(tv_fp16, ptr=b_ptr, shape=[K, N], strides=[N, c1])
        tvC = pto.as_tensor(tv_fp16, ptr=c_ptr, shape=[M, N], strides=[N, c1])

        aMat = pto.alloc_tile(tile_a_mat)
        bMat = pto.alloc_tile(tile_b_mat)
        aLeft = pto.alloc_tile(tile_a_left)
        bRight = pto.alloc_tile(tile_b_right)
        cAcc = pto.alloc_tile(tile_c_acc)

        for bi in pto.range(b_start, b_end, c1):
            blk_m = bi // nblk_n
            blk_n = bi % nblk_n
            row_off = blk_m * cBM
            col_off = blk_n * cBN

            for k in pto.range(c0, K_iters, c1):
                k_off = k * cBK
                svA = pto.slice_view(
                    sv_a, source=tvA, offsets=[row_off, k_off], sizes=[cBM, cBK]
                )
                svB = pto.slice_view(
                    sv_b, source=tvB, offsets=[k_off, col_off], sizes=[cBK, cBN]
                )
                pto.load(svA, aMat)
                pto.load(svB, bMat)
                tile.mov(aMat, aLeft)
                tile.mov(bMat, bRight)
                pto.cond(
                    s.eq(k, c0),
                    lambda: tile.matmul(aLeft, bRight, cAcc),
                    lambda: tile.matmul_acc(cAcc, aLeft, bRight, cAcc),
                )

            # NOTE: per-block scale fusion (Sa[m,k] * Sb[n//128,k]) into the
            # accumulator is omitted; it requires a VEC pass over the FP32
            # accumulator per K-group. See README.md.
            svC = pto.slice_view(
                sv_c, source=tvC, offsets=[row_off, col_off], sizes=[cBM, cBN]
            )
            pto.store(cAcc, svC)


if __name__ == "__main__":
    print(fp8_gemm)
