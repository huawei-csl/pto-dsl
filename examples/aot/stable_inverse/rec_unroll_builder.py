# pyright: reportUndefinedVariable=false
import argparse

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const
SUPPORTED_MATRIX_SIZES = (16, 32, 64, 128)


def make_meta_data(matrix_size: int):
    def meta_data():
        in_dtype = pto.float16
        out_dtype = pto.float32
        i32 = pto.int32

        in_ptr_type = pto.PtrType(in_dtype)
        out_ptr_type = pto.PtrType(out_dtype)

        in_tensor_type = pto.TensorType(rank=3, dtype=in_dtype)
        out_tensor_type = pto.TensorType(rank=3, dtype=out_dtype)
        i_neg_tensor_type = pto.TensorType(rank=2, dtype=in_dtype)

        in_subtensor = pto.SubTensorType(shape=[matrix_size, matrix_size], dtype=in_dtype)
        out_subtensor = pto.SubTensorType(shape=[matrix_size, matrix_size], dtype=out_dtype)
        i_neg_subtensor = pto.SubTensorType(
            shape=[matrix_size, matrix_size], dtype=in_dtype
        )

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
            "i_neg_tensor_type": i_neg_tensor_type,
            "in_subtensor": in_subtensor,
            "out_subtensor": out_subtensor,
            "i_neg_subtensor": i_neg_subtensor,
            "l1_tile_type": l1_tile_type,
            "l0a_tile_type": l0a_tile_type,
            "l0b_tile_type": l0b_tile_type,
            "l0c_tile_type": l0c_tile_type,
        }

    return meta_data


def _emit_single_tile_autosync(
    x_l1,
    y_l1,
    i_l1,
    i_neg_l1,
    a_l0,
    b_l0,
    c_l0,
    matrix_half,
    matrix_quarter,
):
    c1 = const(1)
    c2 = const(2)
    c4 = const(4)
    c8 = const(8)
    c16 = const(16)
    c32 = const(32)

    # X <- I - M, Y <- M @ M
    tile.mov(y_l1, a_l0)
    tile.mov(y_l1, b_l0)
    tile.matmul(a_l0, b_l0, c_l0)
    tile.mov(c_l0, y_l1)

    tile.mov(i_neg_l1, b_l0)
    tile.matmul(a_l0, b_l0, c_l0)

    tile.mov(i_neg_l1, a_l0)
    tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)
    tile.mov(c_l0, x_l1)

    def run_iteration(loop_i):
        tile.mov(x_l1, a_l0)
        tile.mov(i_l1, b_l0)
        tile.matmul(a_l0, b_l0, c_l0)

        tile.mov(y_l1, b_l0)
        tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)

        with pto.if_context(loop_i < matrix_quarter):
            tile.mov(c_l0, x_l1)
            tile.mov(y_l1, a_l0)
            tile.matmul(a_l0, b_l0, c_l0)
            tile.mov(c_l0, y_l1)

    for loop_i in (c1, c2, c4, c8, c16, c32):
        with pto.if_context(loop_i < matrix_half):
            run_iteration(loop_i)


def _emit_single_tile_manualsync(
    x_l1,
    y_l1,
    i_l1,
    i_neg_l1,
    a_l0,
    b_l0,
    c_l0,
    matrix_half,
    matrix_quarter,
):
    c1 = const(1)
    c2 = const(2)
    c4 = const(4)
    c8 = const(8)
    c16 = const(16)
    c32 = const(32)

    tile.mov(y_l1, a_l0)
    pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
    tile.mov(y_l1, b_l0)
    pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
    tile.matmul(a_l0, b_l0, c_l0)
    pto.record_wait_pair("MATMUL", "MOV_V2M", event_id=0)
    tile.mov(c_l0, y_l1)
    pto.record_wait_pair("MOV_V2M", "MOV_M2L", event_id=0)

    tile.mov(i_neg_l1, b_l0)
    pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
    tile.matmul(a_l0, b_l0, c_l0)
    pto.record_wait_pair("MATMUL", "MOV_M2L", event_id=0)

    tile.mov(i_neg_l1, a_l0)
    pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
    tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)
    pto.record_wait_pair("MATMUL", "MOV_V2M", event_id=0)
    tile.mov(c_l0, x_l1)
    pto.record_wait_pair("MOV_V2M", "MATMUL", event_id=0)

    def run_iteration(loop_i):
        tile.mov(x_l1, a_l0)
        tile.mov(i_l1, b_l0)
        pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
        tile.matmul(a_l0, b_l0, c_l0)
        pto.record_wait_pair("MATMUL", "MOV_M2L", event_id=0)

        tile.mov(y_l1, b_l0)
        pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
        tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)

        with pto.if_context(loop_i < matrix_quarter):
            pto.record_wait_pair("MATMUL", "MOV_V2M", event_id=0)
            tile.mov(c_l0, x_l1)
            pto.record_wait_pair("MOV_V2M", "MOV_M2L", event_id=0)
            tile.mov(y_l1, a_l0)
            pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
            tile.matmul(a_l0, b_l0, c_l0)
            pto.record_wait_pair("MATMUL", "MOV_V2M", event_id=0)
            tile.mov(c_l0, y_l1)
            pto.record_wait_pair("MOV_V2M", "MATMUL", event_id=0)

    for loop_i in (c1, c2, c4, c8, c16, c32):
        with pto.if_context(loop_i < matrix_half):
            run_iteration(loop_i)


def _build_kernel_impl(manual_sync: bool, matrix_size: int, kernel_name: str):
    def tri_inv_rec_unroll_fp16(
        out_ptr: "out_ptr_type",
        in_ptr: "in_ptr_type",
        i_neg_ptr: "in_ptr_type",
        matrix_size_i32: "i32",
        num_matrices_i32: "i32",
        num_bsnd_heads_i32: "i32",
    ) -> None:
        with pto.cube_section():
            c0 = const(0)
            c1 = const(1)
            matrix_size_c = const(matrix_size)
            matrix_size_sq = const(matrix_size * matrix_size)
            matrix_half = const(matrix_size // 2)
            matrix_quarter = const(matrix_size // 4)

            # Keep ABI identical to the C++ kernel. matrix_size is specialized by builder.
            _ = matrix_size_i32
            _ = num_bsnd_heads_i32

            num_matrices = s.index_cast(num_matrices_i32)
            bid = s.index_cast(pto.get_block_idx())
            num_blocks = s.index_cast(pto.get_block_num())

            tv_m = pto.as_tensor(
                in_tensor_type,
                ptr=in_ptr,
                shape=[num_matrices, matrix_size_c, matrix_size_c],
                strides=[matrix_size_sq, matrix_size_c, c1],
            )
            tv_out = pto.as_tensor(
                out_tensor_type,
                ptr=out_ptr,
                shape=[num_matrices, matrix_size_c, matrix_size_c],
                strides=[matrix_size_sq, matrix_size_c, c1],
            )
            tv_i_neg = pto.as_tensor(
                i_neg_tensor_type,
                ptr=i_neg_ptr,
                shape=[matrix_size_c, matrix_size_c],
                strides=[matrix_size_c, c1],
            )

            x_l1 = pto.alloc_tile(l1_tile_type)
            y_l1 = pto.alloc_tile(l1_tile_type)
            i_l1 = pto.alloc_tile(l1_tile_type)
            i_neg_l1 = pto.alloc_tile(l1_tile_type)

            a_l0 = pto.alloc_tile(l0a_tile_type)
            b_l0 = pto.alloc_tile(l0b_tile_type)
            c_l0 = pto.alloc_tile(l0c_tile_type)

            sv_i_neg = pto.slice_view(
                i_neg_subtensor,
                source=tv_i_neg,
                offsets=[c0, c0],
                sizes=[matrix_size_c, matrix_size_c],
            )
            pto.load(sv_i_neg, i_neg_l1)
            if manual_sync:
                pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)

            # Build identity in L1 once:
            # I = (-I) @ (-I)
            tile.mov(i_neg_l1, a_l0)
            tile.mov(i_neg_l1, b_l0)
            if manual_sync:
                pto.record_wait_pair("MOV_M2L", "MATMUL", event_id=0)
            tile.matmul(a_l0, b_l0, c_l0)
            if manual_sync:
                pto.record_wait_pair("MATMUL", "MOV_V2M", event_id=0)
            tile.mov(c_l0, i_l1)
            if manual_sync:
                pto.record_wait_pair("MOV_V2M", "MOV_M2L", event_id=0)

            for matrix_idx in pto.range(bid, num_matrices, num_blocks):
                sv_m = pto.slice_view(
                    in_subtensor,
                    source=tv_m,
                    offsets=[matrix_idx, c0, c0],
                    sizes=[c1, matrix_size_c, matrix_size_c],
                )
                sv_out = pto.slice_view(
                    out_subtensor,
                    source=tv_out,
                    offsets=[matrix_idx, c0, c0],
                    sizes=[c1, matrix_size_c, matrix_size_c],
                )

                pto.load(sv_m, y_l1)
                if manual_sync:
                    pto.record_wait_pair("LOAD", "MOV_M2L", event_id=0)

                if manual_sync:
                    _emit_single_tile_manualsync(
                        x_l1,
                        y_l1,
                        i_l1,
                        i_neg_l1,
                        a_l0,
                        b_l0,
                        c_l0,
                        matrix_half,
                        matrix_quarter,
                    )
                    pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
                else:
                    _emit_single_tile_autosync(
                        x_l1,
                        y_l1,
                        i_l1,
                        i_neg_l1,
                        a_l0,
                        b_l0,
                        c_l0,
                        matrix_half,
                        matrix_quarter,
                    )

                pto.store(c_l0, sv_out)
                if manual_sync:
                    pto.record_wait_pair("STORE_ACC", "LOAD", event_id=0)

    tri_inv_rec_unroll_fp16.__name__ = kernel_name
    return to_ir_module(meta_data=make_meta_data(matrix_size))(tri_inv_rec_unroll_fp16)


def build_kernel(manual_sync: bool, matrix_size: int, kernel_name: str):
    return _build_kernel_impl(manual_sync, matrix_size, kernel_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Emit explicit sync events instead of using ptoas auto insert-sync.",
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
    kernel_name = args.kernel_name or f"tri_inv_rec_unroll_fp16_{args.matrix_size}"
    module = build_kernel(args.manual_sync, args.matrix_size, kernel_name)
    print(module)
