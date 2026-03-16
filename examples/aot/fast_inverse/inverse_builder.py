from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const
MAX_MATRIX_SIZE = 128
SUPPORTED_SIZES = (16, 32, 64, 96, 128)


def meta_data():
    in_dtype = pto.float16
    out_dtype = pto.float32
    ptr_in = pto.PtrType(in_dtype)
    ptr_out = pto.PtrType(out_dtype)
    i32 = pto.int32

    tensor_in = pto.TensorType(rank=2, dtype=in_dtype)
    tensor_out = pto.TensorType(rank=2, dtype=out_dtype)
    subtensor_in = pto.SubTensorType(shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE], dtype=in_dtype)
    subtensor_out = pto.SubTensorType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE], dtype=out_dtype
    )

    tile_mat = pto.TileBufType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE],
        valid_shape=[-1, -1],
        dtype=in_dtype,
        memory_space="MAT",
    )
    tile_left = pto.TileBufType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE],
        valid_shape=[-1, -1],
        dtype=in_dtype,
        memory_space="LEFT",
    )
    tile_right = pto.TileBufType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE],
        valid_shape=[-1, -1],
        dtype=in_dtype,
        memory_space="RIGHT",
    )
    tile_acc = pto.TileBufType(
        shape=[MAX_MATRIX_SIZE, MAX_MATRIX_SIZE],
        valid_shape=[-1, -1],
        dtype=out_dtype,
        memory_space="ACC",
    )

    return {
        "ptr_in": ptr_in,
        "ptr_out": ptr_out,
        "i32": i32,
        "tensor_in": tensor_in,
        "tensor_out": tensor_out,
        "subtensor_in": subtensor_in,
        "subtensor_out": subtensor_out,
        "tile_mat": tile_mat,
        "tile_left": tile_left,
        "tile_right": tile_right,
        "tile_acc": tile_acc,
    }


def _emit_tri_inv_kernel(
    out_ptr,
    m_ptr,
    i_neg_ptr,
    matrix_size_i32,
    max_block_size_i32,
    *,
    manual_sync,
):
    c0 = const(0)
    c1 = const(1)
    c2 = const(2)
    c_max = const(MAX_MATRIX_SIZE)

    matrix_size = s.index_cast(matrix_size_i32)
    max_block_size = s.index_cast(max_block_size_i32)
    bid = s.index_cast(pto.get_block_idx())
    num_blocks = s.index_cast(pto.get_block_num())

    def sync(record_op, wait_op):
        if manual_sync:
            pto.record_wait_pair(record_op, wait_op, event_id=0)

    with pto.cube_section():
        with pto.if_context(matrix_size > c0):
            with pto.if_context(matrix_size <= c_max):
                total_rows = num_blocks * matrix_size
                row_offset = bid * matrix_size

                tv_m = pto.as_tensor(
                    tensor_in,
                    ptr=m_ptr,
                    shape=[total_rows, matrix_size],
                    strides=[matrix_size, c1],
                )
                tv_i_neg = pto.as_tensor(
                    tensor_in,
                    ptr=i_neg_ptr,
                    shape=[matrix_size, matrix_size],
                    strides=[matrix_size, c1],
                )
                tv_out = pto.as_tensor(
                    tensor_out,
                    ptr=out_ptr,
                    shape=[total_rows, matrix_size],
                    strides=[matrix_size, c1],
                )

                sv_m = pto.slice_view(
                    subtensor_in,
                    source=tv_m,
                    offsets=[row_offset, c0],
                    sizes=[matrix_size, matrix_size],
                )
                sv_i_neg = pto.slice_view(
                    subtensor_in,
                    source=tv_i_neg,
                    offsets=[c0, c0],
                    sizes=[matrix_size, matrix_size],
                )
                sv_out = pto.slice_view(
                    subtensor_out,
                    source=tv_out,
                    offsets=[row_offset, c0],
                    sizes=[matrix_size, matrix_size],
                )

                x_l1 = pto.alloc_tile(tile_mat, valid_row=matrix_size, valid_col=matrix_size)
                y_l1 = pto.alloc_tile(tile_mat, valid_row=matrix_size, valid_col=matrix_size)
                i_l1 = pto.alloc_tile(tile_mat, valid_row=matrix_size, valid_col=matrix_size)

                a_l0 = pto.alloc_tile(tile_left, valid_row=matrix_size, valid_col=matrix_size)
                b_l0 = pto.alloc_tile(tile_right, valid_row=matrix_size, valid_col=matrix_size)
                c_l0 = pto.alloc_tile(tile_acc, valid_row=matrix_size, valid_col=matrix_size)

                pto.load(sv_m, y_l1)
                pto.load(sv_i_neg, x_l1)
                sync("LOAD", "MOV_M2L")

                tile.mov(y_l1, a_l0)
                tile.mov(y_l1, b_l0)
                sync("MOV_M2L", "MATMUL")

                tile.matmul(a_l0, b_l0, c_l0)  # Y = M @ M
                sync("MATMUL", "MOV_M2L")
                tile.mov(c_l0, y_l1)
                sync("MOV_M2L", "MATMUL")

                tile.mov(x_l1, b_l0)
                sync("MOV_M2L", "MATMUL")
                tile.matmul(a_l0, b_l0, c_l0)  # C = Y @ I_neg
                sync("MATMUL", "MOV_M2L")

                tile.mov(x_l1, a_l0)
                sync("MOV_M2L", "MATMUL")
                tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)  # C = C + I_neg @ I_neg
                sync("MATMUL", "MOV_M2L")

                tile.mov(c_l0, x_l1)  # X
                sync("MOV_M2L", "MATMUL")

                tile.matmul(a_l0, b_l0, c_l0)  # I
                sync("MATMUL", "MOV_M2L")
                tile.mov(c_l0, i_l1)
                sync("MOV_M2L", "MATMUL")

                # Emulate C++ loop i = 1; i < max_block_size; i *= 2.
                for i_value in (1, 2, 4, 8, 16, 32, 64):
                    i_const = const(i_value)
                    with pto.if_context(i_const < max_block_size):
                        tile.mov(x_l1, a_l0)
                        tile.mov(i_l1, b_l0)
                        sync("MOV_M2L", "MATMUL")

                        tile.matmul(a_l0, b_l0, c_l0)  # C = X
                        sync("MATMUL", "MOV_M2L")

                        tile.mov(y_l1, b_l0)
                        sync("MOV_M2L", "MATMUL")
                        tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)  # C = X + X @ Y
                        sync("MATMUL", "MOV_M2L")

                        with pto.if_context(i_const < (max_block_size // c2)):
                            tile.mov(c_l0, x_l1)
                            tile.mov(y_l1, a_l0)
                            sync("MOV_M2L", "MATMUL")

                            tile.matmul(a_l0, b_l0, c_l0)  # Y = Y @ Y
                            sync("MATMUL", "MOV_M2L")
                            tile.mov(c_l0, y_l1)
                            sync("MOV_M2L", "MATMUL")

                if manual_sync:
                    pto.record_wait_pair("MATMUL", "STORE_ACC", event_id=0)
                pto.store(c_l0, sv_out)
                if manual_sync:
                    pto.record_wait_pair("STORE_ACC", "MATMUL", event_id=0)


@to_ir_module(meta_data=meta_data)
def tri_inv_trick_fp16_autosync(
    out_ptr: "ptr_out",
    m_ptr: "ptr_in",
    i_neg_ptr: "ptr_in",
    matrix_size_i32: "i32",
    max_block_size_i32: "i32",
) -> None:
    _emit_tri_inv_kernel(
        out_ptr,
        m_ptr,
        i_neg_ptr,
        matrix_size_i32,
        max_block_size_i32,
        manual_sync=False,
    )


@to_ir_module(meta_data=meta_data)
def tri_inv_trick_fp16_manualsync(
    out_ptr: "ptr_out",
    m_ptr: "ptr_in",
    i_neg_ptr: "ptr_in",
    matrix_size_i32: "i32",
    max_block_size_i32: "i32",
) -> None:
    _emit_tri_inv_kernel(
        out_ptr,
        m_ptr,
        i_neg_ptr,
        matrix_size_i32,
        max_block_size_i32,
        manual_sync=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Emit explicit synchronization ops instead of relying on --enable-insert-sync.",
    )
    args = parser.parse_args()

    module = tri_inv_trick_fp16_manualsync if args.manual_sync else tri_inv_trick_fp16_autosync
    print(module)
