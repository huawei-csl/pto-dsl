# pyright: reportUndefinedVariable=false
import argparse

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const
MAX_MATRIX_SIZE = 128


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

    in_subtensor = pto.SubTensorType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE], dtype=in_dtype
    )
    out_subtensor = pto.SubTensorType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE], dtype=out_dtype
    )

    l1_tile_type = pto.TileBufType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE],
        valid_shape=[-1, -1],
        dtype=in_dtype,
        memory_space="MAT",
    )
    l0a_tile_type = pto.TileBufType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE],
        valid_shape=[-1, -1],
        dtype=in_dtype,
        memory_space="LEFT",
    )
    l0b_tile_type = pto.TileBufType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE],
        valid_shape=[-1, -1],
        dtype=in_dtype,
        memory_space="RIGHT",
    )
    l0c_tile_type = pto.TileBufType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE],
        valid_shape=[-1, -1],
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


def build_kernel(manual_sync: bool):
    @to_ir_module(meta_data=meta_data)
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
            c128 = const(MAX_MATRIX_SIZE)

            matrix_size = s.index_cast(matrix_size_i32)
            max_block_size = s.index_cast(max_block_size_i32)
            block_idx = s.index_cast(pto.get_block_idx())
            num_blocks = s.index_cast(pto.get_block_num())

            with pto.if_context(matrix_size <= c128):
                total_rows = num_blocks * matrix_size
                row_offset = block_idx * matrix_size

                tv_m = pto.as_tensor(
                    in_tensor_type,
                    ptr=in_ptr,
                    shape=[total_rows, matrix_size],
                    strides=[matrix_size, c1],
                )
                tv_out = pto.as_tensor(
                    out_tensor_type,
                    ptr=out_ptr,
                    shape=[total_rows, matrix_size],
                    strides=[matrix_size, c1],
                )
                tv_i_neg = pto.as_tensor(
                    in_tensor_type,
                    ptr=i_neg_ptr,
                    shape=[matrix_size, matrix_size],
                    strides=[matrix_size, c1],
                )

                sv_m = pto.slice_view(
                    in_subtensor,
                    source=tv_m,
                    offsets=[row_offset, c0],
                    sizes=[matrix_size, matrix_size],
                )
                sv_i_neg = pto.slice_view(
                    in_subtensor,
                    source=tv_i_neg,
                    offsets=[c0, c0],
                    sizes=[matrix_size, matrix_size],
                )
                sv_out = pto.slice_view(
                    out_subtensor,
                    source=tv_out,
                    offsets=[row_offset, c0],
                    sizes=[matrix_size, matrix_size],
                )

                x_l1 = pto.alloc_tile(
                    l1_tile_type, valid_row=matrix_size, valid_col=matrix_size
                )
                y_l1 = pto.alloc_tile(
                    l1_tile_type, valid_row=matrix_size, valid_col=matrix_size
                )
                i_l1 = pto.alloc_tile(
                    l1_tile_type, valid_row=matrix_size, valid_col=matrix_size
                )
                a_l0 = pto.alloc_tile(
                    l0a_tile_type, valid_row=matrix_size, valid_col=matrix_size
                )
                b_l0 = pto.alloc_tile(
                    l0b_tile_type, valid_row=matrix_size, valid_col=matrix_size
                )
                c_l0 = pto.alloc_tile(
                    l0c_tile_type, valid_row=matrix_size, valid_col=matrix_size
                )

                def sync(record_op, wait_op):
                    if manual_sync:
                        pto.record_wait_pair(record_op, wait_op, event_id=0)

                def spill_acc_to_mat(dst_l1):
                    sync("MATMUL", "MOV_V2M")
                    tile.mov(c_l0, dst_l1)
                    sync("MOV_V2M", "MOV_M2L")

                pto.load(sv_m, y_l1)
                pto.load(sv_i_neg, x_l1)
                sync("LOAD", "MOV_M2L")

                tile.mov(y_l1, a_l0)
                sync("MOV_M2L", "MATMUL")

                tile.mov(y_l1, b_l0)
                sync("MOV_M2L", "MATMUL")

                tile.matmul(a_l0, b_l0, c_l0)
                spill_acc_to_mat(y_l1)

                tile.mov(x_l1, b_l0)
                sync("MOV_M2L", "MATMUL")
                tile.matmul(a_l0, b_l0, c_l0)
                sync("MATMUL", "MOV_M2L")

                tile.mov(x_l1, a_l0)
                sync("MOV_M2L", "MATMUL")
                tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)
                spill_acc_to_mat(x_l1)

                tile.matmul(a_l0, b_l0, c_l0)
                spill_acc_to_mat(i_l1)

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
                        spill_acc_to_mat(x_l1)
                        tile.mov(y_l1, a_l0)
                        sync("MOV_M2L", "MATMUL")
                        tile.matmul(a_l0, b_l0, c_l0)
                        spill_acc_to_mat(y_l1)

                # Mirror C++ `for (i = 1; i < max_block_size; i *= 2)`.
                # Using pto.range(1, max_block_size, 1) adds many no-op
                # iterations that still perturb generated sync scheduling.
                for loop_i in (c1, c2, c4, c8):
                    with pto.if_context(loop_i < max_block_size):
                        run_iteration(loop_i)

                sync("MATMUL", "STORE_ACC")
                pto.store(c_l0, sv_out)

    return tri_inv_trick_fp16


tri_inv_trick_fp16_autosync = build_kernel(manual_sync=False)
tri_inv_trick_fp16_manualsync = build_kernel(manual_sync=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Emit explicit record/wait events instead of relying on --enable-insert-sync.",
    )
    args = parser.parse_args()
    module = tri_inv_trick_fp16_manualsync if args.manual_sync else tri_inv_trick_fp16_autosync
    print(module)