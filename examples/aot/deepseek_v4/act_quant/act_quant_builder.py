"""PTO DSL port of TileLang act_quant kernel.

Original (GPU): block-wise FP8 quantization, BF16 -> FP8(e4m3) with FP32
or E8M0 per-block scale. inplace=True does fused quant+dequant back to BF16.

NPU port: BF16/FP8 are not native to PTO; we use FP16 input -> int8 output
with FP32 per-block scale. The shape contract matches the original:

    X: [M, N]   fp16
    Y: [M, N]   int8     (quantized) or fp16 (inplace dequant)
    S: [M, N/B] fp32     per-block reciprocal scale

`block_size` is the per-row group size on the K-dim (last axis).
"""

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

BLOCK_SIZE = 128  # K-dim group size; matches GPU `block_size`
BLK_M = 32  # rows per tile (matches GPU `blk_m`)
INT8_MAX = 127.0


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

    row_cfg = pto.TileBufConfig()
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
def act_quant(
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
    inv_max = const(1.0 / INT8_MAX, s.float32)

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
        # Scale layout is COL-MAJOR in memory (strides=[1, M]) so that a
        # [BLK_M, 1] col-major amax tile maps to a contiguous 32-element
        # write at offset `blk_n * M + row_off`.
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
                tile.cvt(tb_x, tb_xf)  # fp16 -> fp32
                tile.abs(tb_xf, tb_abs)  # |x|
                tile.row_max(tb_abs, tb_tmp, tb_amax)  # amax per row
                # scale = amax / 127  (fp32 reciprocal-style scale)
                tile.muls(tb_amax, inv_max, tb_amax)
                # y = x / scale, then cvt -> fp16 -> i8 (NPU has no direct
                # fp32->i8 cvt; routing through fp16 matches the existing
                # quant_dynamic_multicore example).
                tile.row_expand_div(tb_xf, tb_amax, tb_xf)
                tile.cvt(tb_xf, tb_x)  # fp32 -> fp16 (reuse tb_x)
                tile.cvt(tb_x, tb_y, rmode="round")  # fp16 -> int8
                pto.store(tb_y, sv_y)
                pto.store(tb_amax, sv_s)


if __name__ == "__main__":
    print(act_quant)
