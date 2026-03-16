# pyright: reportUndefinedVariable=false
import argparse

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const
MAX_MATRIX_SIZE = 128
SUPPORTED_MATRIX_SIZES = (16, 32, 64, 96, 128)


def make_meta_data(matrix_size: int):
    def meta_data():
        # Match the hand-written kernel:
        # - MAT/LEFT/RIGHT tiles are fp16
        # - ACC and global output are fp32
        # This enables legal TMOV Acc(fp32) -> Mat(fp16) lowering.
        in_dtype = pto.float16
        out_dtype = pto.float32
        i32 = pto.int32

        in_ptr_type = pto.PtrType(in_dtype)
        out_ptr_type = pto.PtrType(out_dtype)

        in_tensor_type = pto.TensorType(rank=2, dtype=in_dtype)
        out_tensor_type = pto.TensorType(rank=2, dtype=out_dtype)
        in_subtensor = pto.SubTensorType(shape=[matrix_size, matrix_size], dtype=in_dtype)
        out_subtensor = pto.SubTensorType(shape=[matrix_size, matrix_size], dtype=out_dtype)
        l1_tile_type = pto.TileBufType(
            shape=[matrix_size, matrix_size],
            valid_shape=[matrix_size, matrix_size],
            dtype=in_dtype,
            memory_space="MAT",
        )
        l0a_tile_type = pto.TileBufType(
            shape=[matrix_size, matrix_size],
            valid_shape=[matrix_size, matrix_size],
            dtype=in_dtype,
            memory_space="LEFT",
        )
        l0b_tile_type = pto.TileBufType(
            shape=[matrix_size, matrix_size],
            valid_shape=[matrix_size, matrix_size],
            dtype=in_dtype,
            memory_space="RIGHT",
        )
        l0c_tile_type = pto.TileBufType(
            shape=[matrix_size, matrix_size],
            valid_shape=[matrix_size, matrix_size],
            dtype=out_dtype,
            memory_space="ACC",
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


def build_kernel(manual_sync: bool, matrix_size: int, kernel_name: str):
    matrix_size_c = const(matrix_size)

    def tri_inv_trick_fp16(
        out_ptr: "out_ptr_type",
        in_ptr: "in_ptr_type",
        i_neg_ptr: "in_ptr_type",
        matrix_size_i32: "i32",
        max_block_size_i32: "i32",
    ) -> None:
        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            c2 = const(2)
            c4 = const(4)
            c8 = const(8)

            max_block_size = s.index_cast(max_block_size_i32)
            block_idx = s.index_cast(pto.get_block_idx())
            num_blocks = s.index_cast(pto.get_block_num())

            total_rows = num_blocks * matrix_size_c
            row_offset = block_idx * matrix_size_c

            # Keep the runtime signature unchanged while emitting
            # compile-time-specialized tile/subtensor types.
            _ = matrix_size_i32

            tv_m = pto.as_tensor(
                in_tensor_type,
                ptr=in_ptr,
                shape=[total_rows, matrix_size_c],
                strides=[matrix_size_c, c1],
            )
            tv_out = pto.as_tensor(
                out_tensor_type,
                ptr=out_ptr,
                shape=[total_rows, matrix_size_c],
                strides=[matrix_size_c, c1],
            )
            tv_i_neg = pto.as_tensor(
                in_tensor_type,
                ptr=i_neg_ptr,
                shape=[matrix_size_c, matrix_size_c],
                strides=[matrix_size_c, c1],
            )

            sv_m = pto.slice_view(
                in_subtensor,
                source=tv_m,
                offsets=[row_offset, c0],
                sizes=[matrix_size_c, matrix_size_c],
            )
            sv_i_neg = pto.slice_view(
                in_subtensor,
                source=tv_i_neg,
                offsets=[c0, c0],
                sizes=[matrix_size_c, matrix_size_c],
            )
            sv_out = pto.slice_view(
                out_subtensor,
                source=tv_out,
                offsets=[row_offset, c0],
                sizes=[matrix_size_c, matrix_size_c],
            )

            x_l1 = pto.alloc_tile(
                l1_tile_type, valid_row=matrix_size_c, valid_col=matrix_size_c
            )
            y_l1 = pto.alloc_tile(
                l1_tile_type, valid_row=matrix_size_c, valid_col=matrix_size_c
            )
            i_l1 = pto.alloc_tile(
                l1_tile_type, valid_row=matrix_size_c, valid_col=matrix_size_c
            )
            a_l0 = pto.alloc_tile(
                l0a_tile_type, valid_row=matrix_size_c, valid_col=matrix_size_c
            )
            b_l0 = pto.alloc_tile(
                l0b_tile_type, valid_row=matrix_size_c, valid_col=matrix_size_c
            )
            c_l0 = pto.alloc_tile(
                l0c_tile_type, valid_row=matrix_size_c, valid_col=matrix_size_c
            )

            def sync(record_op, wait_op):
                if manual_sync:
                    pto.record_wait_pair(record_op, wait_op, event_id=0)

            pto.load(sv_m, y_l1)
            pto.load(sv_i_neg, x_l1)
            sync("LOAD", "MOV_M2L")

            tile.mov(y_l1, a_l0)
            sync("MOV_M2L", "MATMUL")

            tile.mov(y_l1, b_l0)
            sync("MOV_M2L", "MATMUL")

            tile.matmul(a_l0, b_l0, c_l0)
            sync("MATMUL", "MOV_V2M")
            tile.mov(c_l0, y_l1)
            sync("MOV_V2M", "MOV_M2L")

            tile.mov(x_l1, b_l0)
            sync("MOV_M2L", "MATMUL")
            tile.matmul(a_l0, b_l0, c_l0)
            sync("MATMUL", "MOV_M2L")

            tile.mov(x_l1, a_l0)
            sync("MOV_M2L", "MATMUL")
            tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)
            sync("MATMUL", "MOV_V2M")
            tile.mov(c_l0, x_l1)
            sync("MOV_V2M", "MATMUL")

            tile.matmul(a_l0, b_l0, c_l0)
            sync("MATMUL", "MOV_V2M")
            tile.mov(c_l0, i_l1)
            sync("MOV_V2M", "MOV_M2L")

            def run_iteration(iter_i):
                tile.mov(x_l1, a_l0)
                tile.mov(i_l1, b_l0)
                sync("MOV_M2L", "MATMUL")
                tile.matmul(a_l0, b_l0, c_l0)
                sync("MATMUL", "MOV_M2L")

                tile.mov(y_l1, b_l0)
                sync("MOV_M2L", "MATMUL")
                tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)

                with pto.if_context(iter_i < (max_block_size // c2)):
                    sync("MATMUL", "MOV_V2M")
                    tile.mov(c_l0, x_l1)
                    sync("MOV_V2M", "MOV_M2L")
                    tile.mov(y_l1, a_l0)
                    sync("MOV_M2L", "MATMUL")
                    tile.matmul(a_l0, b_l0, c_l0)
                    sync("MATMUL", "MOV_V2M")
                    tile.mov(c_l0, y_l1)
                    sync("MOV_V2M", "MOV_M2L")

            # Mirror C++ `for (i = 1; i < max_block_size; i *= 2)`.
            # Using pto.range(1, max_block_size, 1) adds many no-op
            # iterations that still perturb generated sync scheduling.
            for loop_i in (c1, c2, c4, c8):
                with pto.if_context(loop_i < max_block_size):
                    run_iteration(loop_i)

            sync("MATMUL", "STORE_ACC")
            pto.store(c_l0, sv_out)

    tri_inv_trick_fp16.__name__ = kernel_name
    return to_ir_module(meta_data=make_meta_data(matrix_size))(tri_inv_trick_fp16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Emit explicit record/wait events instead of relying on --enable-insert-sync.",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        choices=SUPPORTED_MATRIX_SIZES,
        default=128,
        help="Compile-time specialized matrix size.",
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        default=None,
        help="Kernel symbol name in emitted module.",
    )
    args = parser.parse_args()
    kernel_name = args.kernel_name or f"tri_inv_trick_fp16_{args.matrix_size}"
    module = build_kernel(args.manual_sync, args.matrix_size, kernel_name)
    print(module)