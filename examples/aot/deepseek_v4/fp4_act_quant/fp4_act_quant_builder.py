"""PTO DSL port of TileLang fp4_act_quant kernel.

Original (GPU): block-wise BF16 -> FP4(e2m1) quantization, FP4_max=6,
power-of-2 (E8M0) per-block scale, block_size=32 on the K-dim.

NPU port: FP4 / BF16 / E8M0 are not native to PTO. We implement the
same algorithm with FP16 inputs and int4-equivalent quantization stored
in int8 (one int4 per byte). For simplicity the output container is
int8 with values in [-7, 7]; per-block scale stays FP32.
"""

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

BLOCK_SIZE = 32  # K-dim group size; matches GPU `fp4_block_size`
BLK_M = 32
FP4_MAX = 6.0


def meta_data():
    fp16 = pto.float16
    fp32 = pto.float32
    i8 = pto.int8
    i32 = pto.int32

    ptr_fp16 = pto.PtrType(fp16)
    ptr_i8 = pto.PtrType(i8)
    ptr_fp32 = pto.PtrType(fp32)

    tv_fp16 = pto.TensorType(rank=2, dtype=fp16)
    tv_i8 = pto.TensorType(rank=2, dtype=i8)
    tv_fp32 = pto.TensorType(rank=2, dtype=fp32)

    sv_fp16 = pto.SubTensorType(shape=[BLK_M, BLOCK_SIZE], dtype=fp16)
    sv_i8 = pto.SubTensorType(shape=[BLK_M, BLOCK_SIZE], dtype=i8)
    sv_scale = pto.SubTensorType(shape=[BLK_M, 1], dtype=fp32)

    col_cfg = pto.TileBufConfig(blayout="ColMajor")

    tile_fp16 = pto.TileBufType(
        shape=[BLK_M, BLOCK_SIZE], dtype=fp16, memory_space="VEC"
    )
    tile_fp32 = pto.TileBufType(
        shape=[BLK_M, BLOCK_SIZE], dtype=fp32, memory_space="VEC"
    )
    tile_i8 = pto.TileBufType(shape=[BLK_M, BLOCK_SIZE], dtype=i8, memory_space="VEC")
    tile_amax = pto.TileBufType(
        shape=[BLK_M, 1], dtype=fp32, memory_space="VEC", config=col_cfg
    )

    return locals()


@to_ir_module(meta_data=meta_data)
def fp4_act_quant(
    x_ptr: "ptr_fp16",
    y_ptr: "ptr_i8",
    s_ptr: "ptr_fp32",
    M_i32: "i32",
    N_i32: "i32",
) -> None:
    c0 = const(0)
    c1 = const(1)
    cBM = const(BLK_M)
    cBK = const(BLOCK_SIZE)
    inv_max = const(1.0 / FP4_MAX, s.float32)

    M = s.index_cast(M_i32)
    N = s.index_cast(N_i32)
    nblk_n = s.ceil_div(N, cBK)

    with pto.vector_section():
        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        num_blocks = pto.get_block_num()
        vid = s.index_cast(cid * sub_bnum + sub_bid)
        ncores = s.index_cast(num_blocks * sub_bnum)

        nblk_m = s.ceil_div(M, cBM)
        total_blocks = nblk_m * nblk_n

        tv_x = pto.as_tensor(tv_fp16, ptr=x_ptr, shape=[M, N], strides=[N, c1])
        tv_y = pto.as_tensor(tv_i8, ptr=y_ptr, shape=[M, N], strides=[N, c1])
        # Scale layout: COL-MAJOR in memory (strides=[1, M]) so the
        # [BLK_M, 1] col-major amax tile is stored contiguously.
        tv_s = pto.as_tensor(tv_fp32, ptr=s_ptr, shape=[M, nblk_n], strides=[c1, M])

        tb_x = pto.alloc_tile(tile_fp16)
        tb_xf = pto.alloc_tile(tile_fp32)
        tb_abs = pto.alloc_tile(tile_fp32)
        tb_tmp = pto.alloc_tile(tile_fp32)
        tb_amax = pto.alloc_tile(tile_amax)
        tb_y = pto.alloc_tile(tile_i8)

        with pto.if_context(vid < total_blocks):
            for bi in pto.range(vid, total_blocks, ncores):
                blk_m = bi // nblk_n
                blk_n = bi % nblk_n
                row_off = blk_m * cBM
                col_off = blk_n * cBK

                sv_x = pto.slice_view(
                    sv_fp16,
                    source=tv_x,
                    offsets=[row_off, col_off],
                    sizes=[cBM, cBK],
                )
                sv_y = pto.slice_view(
                    sv_i8,
                    source=tv_y,
                    offsets=[row_off, col_off],
                    sizes=[cBM, cBK],
                )
                sv_s = pto.slice_view(
                    sv_scale,
                    source=tv_s,
                    offsets=[row_off, blk_n],
                    sizes=[cBM, c1],
                )

                pto.load(sv_x, tb_x)
                tile.cvt(tb_x, tb_xf)
                tile.abs(tb_xf, tb_abs)
                tile.row_max(tb_abs, tb_tmp, tb_amax)
                tile.muls(tb_amax, inv_max, tb_amax)  # scale = amax / 6
                tile.row_expand_div(tb_xf, tb_amax, tb_xf)
                # fp32 -> fp16 -> int8 (NPU has no direct fp32->i8 cvt).
                tile.cvt(tb_xf, tb_x)
                tile.cvt(tb_x, tb_y, rmode="round")
                pto.store(tb_y, sv_y)
                pto.store(tb_amax, sv_s)


if __name__ == "__main__":
    print(fp4_act_quant)
