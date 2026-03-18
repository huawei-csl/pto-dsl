# pyright: reportUndefinedVariable=false
import argparse

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const
SUPPORTED_MATRIX_SIZES = (16, 32, 64, 96, 128)


def make_meta_data(n: int):  # input matrix size
    def meta_data():
        in_dtype = pto.float16
        out_dtype = pto.float32
        i32 = pto.int32

        in_ptr_type = pto.PtrType(in_dtype)
        out_ptr_type = pto.PtrType(out_dtype)
        in_tensor_type = pto.TensorType(rank=2, dtype=in_dtype)
        out_tensor_type = pto.TensorType(rank=2, dtype=out_dtype)
        in_subtensor = pto.SubTensorType(shape=[n, n], dtype=in_dtype)
        out_subtensor = pto.SubTensorType(shape=[n, n], dtype=out_dtype)
        l1_tile_type = pto.TileBufType(
            shape=[n, n], valid_shape=[n, n], dtype=in_dtype, memory_space="MAT"
        )
        l0a_tile_type = pto.TileBufType(
            shape=[n, n], valid_shape=[n, n], dtype=in_dtype, memory_space="LEFT"
        )
        l0b_tile_type = pto.TileBufType(
            shape=[n, n], valid_shape=[n, n], dtype=in_dtype, memory_space="RIGHT"
        )
        l0c_tile_type = pto.TileBufType(
            shape=[n, n], valid_shape=[n, n], dtype=out_dtype, memory_space="ACC"
        )

        return {
            "in_ptr_type": in_ptr_type,
            "out_ptr_type": out_ptr_type,
            "i32": i32,
            "in_tensor_type": in_tensor_type,
            "out_tensor_type": out_tensor_type,
            "in_subtensor": in_subtensor,
            "out_subtensor": out_subtensor,
            "l1_tile_type": l1_tile_type,
            "l0a_tile_type": l0a_tile_type,
            "l0b_tile_type": l0b_tile_type,
            "l0c_tile_type": l0c_tile_type,
        }

    return meta_data


def build_kernel(matrix_size: int):
    @to_ir_module(meta_data=make_meta_data(matrix_size))
    def tri_inv_trick_fp16(
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

            log2_blocksize = s.index_cast(log2_blocksize_i32)
            block_idx = s.index_cast(pto.get_block_idx())
            num_blocks = s.index_cast(pto.get_block_num())

            total_rows = num_blocks * n_c
            row_offset = block_idx * n_c

            tv_m = pto.as_tensor(
                in_tensor_type, ptr=in_ptr, shape=[total_rows, n_c], strides=[n_c, c1]
            )
            tv_out = pto.as_tensor(
                out_tensor_type, ptr=out_ptr, shape=[total_rows, n_c], strides=[n_c, c1]
            )
            tv_i_neg = pto.as_tensor(
                in_tensor_type, ptr=i_neg_ptr, shape=[n_c, n_c], strides=[n_c, c1]
            )

            sv_m = pto.slice_view(
                in_subtensor, source=tv_m, offsets=[row_offset, c0], sizes=[n_c, n_c]
            )
            sv_i_neg = pto.slice_view(
                in_subtensor, source=tv_i_neg, offsets=[c0, c0], sizes=[n_c, n_c]
            )
            sv_out = pto.slice_view(
                out_subtensor, source=tv_out, offsets=[row_offset, c0], sizes=[n_c, n_c]
            )

            x_l1 = pto.alloc_tile(l1_tile_type)
            y_l1 = pto.alloc_tile(l1_tile_type)
            i_l1 = pto.alloc_tile(l1_tile_type)
            a_l0 = pto.alloc_tile(l0a_tile_type)
            b_l0 = pto.alloc_tile(l0b_tile_type)
            c_l0 = pto.alloc_tile(l0c_tile_type)

            pto.load(sv_m, y_l1)
            pto.load(sv_i_neg, x_l1)

            tile.mov(y_l1, a_l0)
            tile.mov(y_l1, b_l0)

            tile.matmul(a_l0, b_l0, c_l0)
            tile.mov(c_l0, y_l1)

            tile.mov(x_l1, b_l0)
            tile.matmul(a_l0, b_l0, c_l0)

            tile.mov(x_l1, a_l0)
            tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)
            tile.mov(c_l0, x_l1)

            tile.matmul(a_l0, b_l0, c_l0)
            tile.mov(c_l0, i_l1)

            for iter_idx in pto.range(c0, log2_blocksize, c1):
                tile.mov(x_l1, a_l0)
                tile.mov(i_l1, b_l0)
                tile.matmul(a_l0, b_l0, c_l0)

                tile.mov(y_l1, b_l0)
                tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)

                with pto.if_context(iter_idx + c1 < log2_blocksize):
                    tile.mov(c_l0, x_l1)
                    tile.mov(y_l1, a_l0)
                    tile.matmul(a_l0, b_l0, c_l0)
                    tile.mov(c_l0, y_l1)

            pto.store(c_l0, sv_out)

    return tri_inv_trick_fp16


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
