# pyright: reportUndefinedVariable=false
import argparse

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const
SUPPORTED_MATRIX_SIZES = (16, 32, 64, 128)


def make_meta_data(n: int):
    h = n // 2

    def meta_data():
        in_dtype = pto.float16
        out_dtype = pto.float32
        i32 = pto.int32

        in_ptr_type = pto.PtrType(in_dtype)
        out_ptr_type = pto.PtrType(out_dtype)
        in_tensor_type = pto.TensorType(rank=2, dtype=in_dtype)
        out_tensor_type = pto.TensorType(rank=2, dtype=out_dtype)

        in_subtensor_h = pto.SubTensorType(shape=[h, h], dtype=in_dtype)
        out_subtensor_h = pto.SubTensorType(shape=[h, h], dtype=out_dtype)

        l1_tile_type = pto.TileBufType(
            shape=[h, h], valid_shape=[h, h], dtype=in_dtype, memory_space="MAT"
        )
        l0a_tile_type = pto.TileBufType(
            shape=[h, h], valid_shape=[h, h], dtype=in_dtype, memory_space="LEFT"
        )
        l0b_tile_type = pto.TileBufType(
            shape=[h, h], valid_shape=[h, h], dtype=in_dtype, memory_space="RIGHT"
        )
        l0c_tile_type = pto.TileBufType(
            shape=[h, h], valid_shape=[h, h], dtype=out_dtype, memory_space="ACC"
        )

        return {
            "in_ptr_type": in_ptr_type,
            "out_ptr_type": out_ptr_type,
            "i32": i32,
            "in_tensor_type": in_tensor_type,
            "out_tensor_type": out_tensor_type,
            "in_subtensor_h": in_subtensor_h,
            "out_subtensor_h": out_subtensor_h,
            "l1_tile_type": l1_tile_type,
            "l0a_tile_type": l0a_tile_type,
            "l0b_tile_type": l0b_tile_type,
            "l0c_tile_type": l0c_tile_type,
        }

    return meta_data


def build_kernel(matrix_size: int):
    assert matrix_size % 2 == 0 and matrix_size >= 16

    @to_ir_module(meta_data=make_meta_data(matrix_size))
    def tri_inv_block2x2_fp16(
        out_ptr: "out_ptr_type",
        in_ptr: "in_ptr_type",
        i_neg_ptr: "in_ptr_type",
        matrix_size_i32: "i32",
        log2_blocksize_i32: "i32",
    ) -> None:
        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            n_c = const(matrix_size)
            h_c = const(matrix_size // 2)

            batch_size = s.index_cast(matrix_size_i32)
            log2_half = s.index_cast(log2_blocksize_i32) - c1
            block_idx = s.index_cast(pto.get_block_idx())
            num_cores = s.index_cast(pto.get_block_num())
            total_rows = batch_size * n_c

            # Persistent-kernel work split: base + remainder.
            base = batch_size // num_cores
            rem = batch_size % num_cores
            lt_rem = s.lt(block_idx, rem)
            min_bid_rem = s.min_u(block_idx, rem)
            b_start = block_idx * base + min_bid_rem
            length = base + s.select(lt_rem, c1, c0)
            b_end = s.min_u(b_start + length, batch_size)

            tv_in = pto.as_tensor(
                in_tensor_type, ptr=in_ptr, shape=[total_rows, n_c], strides=[n_c, c1]
            )
            tv_out = pto.as_tensor(
                out_tensor_type, ptr=out_ptr, shape=[total_rows, n_c], strides=[n_c, c1]
            )
            tv_i_neg = pto.as_tensor(
                in_tensor_type, ptr=i_neg_ptr, shape=[h_c, h_c], strides=[h_c, c1]
            )
            sv_i_neg = pto.slice_view(
                in_subtensor_h, source=tv_i_neg, offsets=[c0, c0], sizes=[h_c, h_c]
            )

            x11_l1 = pto.alloc_tile(l1_tile_type)
            y11_l1 = pto.alloc_tile(l1_tile_type)
            x22_l1 = pto.alloc_tile(l1_tile_type)
            y22_l1 = pto.alloc_tile(l1_tile_type)
            a21_l1 = pto.alloc_tile(l1_tile_type)
            neg_i_l1 = pto.alloc_tile(l1_tile_type)
            pos_i_l1 = pto.alloc_tile(l1_tile_type)
            tmp_l1 = pto.alloc_tile(l1_tile_type)

            a_l0 = pto.alloc_tile(l0a_tile_type)
            b_l0 = pto.alloc_tile(l0b_tile_type)
            c_l0 = pto.alloc_tile(l0c_tile_type)

            # Build +/- identity tiles for half-size blocks.
            # Also seed x11 = x22 = I for the recurrence below.
            pto.load(sv_i_neg, neg_i_l1)

            for b_idx in pto.range(b_start, b_end, c1):
                row_offset = b_idx * n_c
                row_offset_h = row_offset + h_c
                sv_a11 = pto.slice_view(
                    in_subtensor_h, source=tv_in, offsets=[row_offset, c0], sizes=[h_c, h_c]
                )
                sv_a21 = pto.slice_view(
                    in_subtensor_h,
                    source=tv_in,
                    offsets=[row_offset_h, c0],
                    sizes=[h_c, h_c],
                )
                sv_a22 = pto.slice_view(
                    in_subtensor_h,
                    source=tv_in,
                    offsets=[row_offset_h, h_c],
                    sizes=[h_c, h_c],
                )
                sv_out11 = pto.slice_view(
                    out_subtensor_h,
                    source=tv_out,
                    offsets=[row_offset, c0],
                    sizes=[h_c, h_c],
                )
                sv_out21 = pto.slice_view(
                    out_subtensor_h,
                    source=tv_out,
                    offsets=[row_offset_h, c0],
                    sizes=[h_c, h_c],
                )
                sv_out22 = pto.slice_view(
                    out_subtensor_h,
                    source=tv_out,
                    offsets=[row_offset_h, h_c],
                    sizes=[h_c, h_c],
                )

                tile.mov(neg_i_l1, a_l0)
                tile.mov(neg_i_l1, b_l0)
                tile.matmul(a_l0, b_l0, c_l0)
                tile.mov(c_l0, pos_i_l1)
                tile.mov(c_l0, x11_l1)  # x11 = I
                tile.mov(c_l0, x22_l1)  # x22 = I

                # Invert (I + A11): start the recurrence with y11 = -A11, x11 = I.
                # The loop then computes x_{k+1} = x_k(I + y_k), y_{k+1} = y_k^2
                # which gives (I + A11)^{-1} after log2_half steps.
                pto.load(sv_a11, y11_l1)
                tile.mov(y11_l1, a_l0)
                tile.mov(neg_i_l1, b_l0)
                tile.matmul(a_l0, b_l0, c_l0)  # c = -A11
                tile.mov(c_l0, y11_l1)          # y11 = -A11

                for iter_idx in pto.range(c0, log2_half, c1):
                    tile.mov(x11_l1, a_l0)
                    tile.mov(pos_i_l1, b_l0)
                    tile.matmul(a_l0, b_l0, c_l0)

                    tile.mov(y11_l1, b_l0)
                    tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)

                    with pto.if_context(iter_idx + c1 < log2_half):
                        tile.mov(c_l0, x11_l1)
                        tile.mov(y11_l1, a_l0)
                        tile.matmul(a_l0, b_l0, c_l0)
                        tile.mov(c_l0, y11_l1)

                tile.mov(c_l0, x11_l1)
                pto.store(c_l0, sv_out11)

                # Invert (I + A22): start with y22 = -A22, x22 = I (already set above).
                pto.load(sv_a22, y22_l1)
                tile.mov(y22_l1, a_l0)
                tile.mov(neg_i_l1, b_l0)
                tile.matmul(a_l0, b_l0, c_l0)  # c = -A22
                tile.mov(c_l0, y22_l1)          # y22 = -A22

                for iter_idx in pto.range(c0, log2_half, c1):
                    tile.mov(x22_l1, a_l0)
                    tile.mov(pos_i_l1, b_l0)
                    tile.matmul(a_l0, b_l0, c_l0)

                    tile.mov(y22_l1, b_l0)
                    tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)

                    with pto.if_context(iter_idx + c1 < log2_half):
                        tile.mov(c_l0, x22_l1)
                        tile.mov(y22_l1, a_l0)
                        tile.matmul(a_l0, b_l0, c_l0)
                        tile.mov(c_l0, y22_l1)

                tile.mov(c_l0, x22_l1)
                pto.store(c_l0, sv_out22)

                # A21 term in block inversion:
                # X21 = - X22 @ A21 @ X11
                pto.load(sv_a21, a21_l1)

                tile.mov(x22_l1, a_l0)
                tile.mov(a21_l1, b_l0)
                tile.matmul(a_l0, b_l0, c_l0)
                tile.mov(c_l0, tmp_l1)

                tile.mov(tmp_l1, a_l0)
                tile.mov(x11_l1, b_l0)
                tile.matmul(a_l0, b_l0, c_l0)
                tile.mov(c_l0, tmp_l1)

                tile.mov(neg_i_l1, a_l0)
                tile.mov(tmp_l1, b_l0)
                tile.matmul(a_l0, b_l0, c_l0)
                pto.store(c_l0, sv_out21)

    return tri_inv_block2x2_fp16


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-size",
        type=int,
        choices=SUPPORTED_MATRIX_SIZES,
        default=64,
        help="Compile-time specialized matrix size.",
    )
    args = parser.parse_args()
    module = build_kernel(args.matrix_size)
    print(module)
