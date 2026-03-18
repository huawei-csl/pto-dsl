# pyright: reportUndefinedVariable=false
import argparse

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const
SUPPORTED_MATRIX_SIZES = (16, 32, 64, 128)


def make_meta_data(n: int):
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


def build_single_buffer_kernel(matrix_size: int):
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

            batch_size = s.index_cast(matrix_size_i32)
            log2_blocksize = s.index_cast(log2_blocksize_i32)
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

            tv_m = pto.as_tensor(
                in_tensor_type, ptr=in_ptr, shape=[total_rows, n_c], strides=[n_c, c1]
            )
            tv_out = pto.as_tensor(
                out_tensor_type, ptr=out_ptr, shape=[total_rows, n_c], strides=[n_c, c1]
            )
            tv_i_neg = pto.as_tensor(
                in_tensor_type, ptr=i_neg_ptr, shape=[n_c, n_c], strides=[n_c, c1]
            )

            sv_i_neg = pto.slice_view(
                in_subtensor, source=tv_i_neg, offsets=[c0, c0], sizes=[n_c, n_c]
            )

            x_l1 = pto.alloc_tile(l1_tile_type)
            y_l1 = pto.alloc_tile(l1_tile_type)
            i_l1 = pto.alloc_tile(l1_tile_type)
            a_l0 = pto.alloc_tile(l0a_tile_type)
            b_l0 = pto.alloc_tile(l0b_tile_type)
            c_l0 = pto.alloc_tile(l0c_tile_type)

            for b_idx in pto.range(b_start, b_end, c1):
                row_offset = b_idx * n_c
                sv_m = pto.slice_view(
                    in_subtensor, source=tv_m, offsets=[row_offset, c0], sizes=[n_c, n_c]
                )
                sv_out = pto.slice_view(
                    out_subtensor, source=tv_out, offsets=[row_offset, c0], sizes=[n_c, n_c]
                )

                # in_ptr carries A = M - I, where M is the dense matrix to invert.
                pto.load(sv_m, y_l1)
                pto.load(sv_i_neg, x_l1)

                tile.mov(y_l1, a_l0)
                tile.mov(y_l1, b_l0)
                tile.matmul(a_l0, b_l0, c_l0)
                tile.mov(c_l0, y_l1)  # y = A @ A

                tile.mov(x_l1, b_l0)
                tile.matmul(a_l0, b_l0, c_l0)  # c = -A

                tile.mov(x_l1, a_l0)
                tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)  # c = I - A
                tile.mov(c_l0, x_l1)  # x = I - A

                tile.matmul(a_l0, b_l0, c_l0)
                tile.mov(c_l0, i_l1)  # i = I

                # Mirrors:
                # for i in range(log2_c - 1):
                #     X, Y = (X + X @ Y, Y @ Y)
                for iter_idx in pto.range(c0, log2_blocksize, c1):
                    tile.mov(x_l1, a_l0)
                    tile.mov(i_l1, b_l0)
                    tile.matmul(a_l0, b_l0, c_l0)

                    tile.mov(y_l1, b_l0)
                    tile.matmul_acc(c_l0, a_l0, b_l0, c_l0)  # x + x @ y

                    with pto.if_context(iter_idx + c1 < log2_blocksize):
                        tile.mov(c_l0, x_l1)
                        tile.mov(y_l1, a_l0)
                        tile.matmul(a_l0, b_l0, c_l0)
                        tile.mov(c_l0, y_l1)  # y = y @ y

                pto.store(c_l0, sv_out)

    return tri_inv_trick_fp16


def build_double_buffer_kernel(matrix_size: int):
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
            c2 = const(2)
            n_c = const(matrix_size)

            batch_size = s.index_cast(matrix_size_i32)
            log2_blocksize = s.index_cast(log2_blocksize_i32)
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

            tv_m = pto.as_tensor(
                in_tensor_type, ptr=in_ptr, shape=[total_rows, n_c], strides=[n_c, c1]
            )
            tv_out = pto.as_tensor(
                out_tensor_type, ptr=out_ptr, shape=[total_rows, n_c], strides=[n_c, c1]
            )
            tv_i_neg = pto.as_tensor(
                in_tensor_type, ptr=i_neg_ptr, shape=[n_c, n_c], strides=[n_c, c1]
            )

            sv_i_neg = pto.slice_view(
                in_subtensor, source=tv_i_neg, offsets=[c0, c0], sizes=[n_c, n_c]
            )

            # Memory footprint (fp16=2B, fp32=4B):
            # - L1: 2*X + 2*Y + I = 5 * (n*n*2B); n=128 -> 160KiB (< 512KiB limit)
            # - L0A: 2*A = 2 * (n*n*2B); n=128 -> 64KiB (at 64KiB limit)
            # - L0B: 2*B = 2 * (n*n*2B); n=128 -> 64KiB (at 64KiB limit)
            # - L0C: 1*C = 1 * (n*n*4B); n=128 -> 64KiB (< 128KiB limit)
            x_l1 = [pto.alloc_tile(l1_tile_type), pto.alloc_tile(l1_tile_type)]
            y_l1 = [pto.alloc_tile(l1_tile_type), pto.alloc_tile(l1_tile_type)]
            i_l1 = pto.alloc_tile(l1_tile_type)
            a_l0 = [pto.alloc_tile(l0a_tile_type), pto.alloc_tile(l0a_tile_type)]
            b_l0 = [pto.alloc_tile(l0b_tile_type), pto.alloc_tile(l0b_tile_type)]
            c_l0 = pto.alloc_tile(l0c_tile_type)

            for b_idx in pto.range(b_start, b_end, c1):
                row_offset = b_idx * n_c
                sv_m = pto.slice_view(
                    in_subtensor, source=tv_m, offsets=[row_offset, c0], sizes=[n_c, n_c]
                )
                sv_out = pto.slice_view(
                    out_subtensor, source=tv_out, offsets=[row_offset, c0], sizes=[n_c, n_c]
                )

                # in_ptr carries A = M - I, where M is the dense matrix to invert.
                pto.load(sv_m, y_l1[0])
                pto.load(sv_i_neg, x_l1[0])

                # Bootstrap from ping buffer 0.
                tile.mov(y_l1[0], a_l0[0])
                tile.mov(y_l1[0], b_l0[0])
                tile.matmul(a_l0[0], b_l0[0], c_l0)
                tile.mov(c_l0, y_l1[0])  # y = A @ A

                tile.mov(x_l1[0], b_l0[0])
                tile.matmul(a_l0[0], b_l0[0], c_l0)  # c = -A

                tile.mov(x_l1[0], a_l0[0])
                tile.matmul_acc(c_l0, a_l0[0], b_l0[0], c_l0)  # c = I - A
                tile.mov(c_l0, x_l1[0])  # x = I - A

                tile.matmul(a_l0[0], b_l0[0], c_l0)
                tile.mov(c_l0, i_l1)  # i = I

                # Mirrors:
                # for i in range(log2_c - 1):
                #     X, Y = (X + X @ Y, Y @ Y)
                for iter_idx in pto.range(c0, log2_blocksize, c1):
                    def run_iter(curr: int, nxt: int):
                        tile.mov(x_l1[curr], a_l0[curr])
                        tile.mov(i_l1, b_l0[curr])
                        tile.matmul(a_l0[curr], b_l0[curr], c_l0)

                        tile.mov(y_l1[curr], b_l0[curr])
                        tile.matmul_acc(c_l0, a_l0[curr], b_l0[curr], c_l0)  # x + x @ y

                        with pto.if_context(iter_idx + c1 < log2_blocksize):
                            tile.mov(c_l0, x_l1[nxt])
                            tile.mov(y_l1[curr], a_l0[curr])
                            tile.matmul(a_l0[curr], b_l0[curr], c_l0)
                            tile.mov(c_l0, y_l1[nxt])  # y = y @ y

                    is_curr0 = (iter_idx % c2) == c0
                    with pto.if_context(is_curr0, has_else=True) as branch:
                        run_iter(0, 1)
                    with branch.else_context():
                        run_iter(1, 0)

                pto.store(c_l0, sv_out)

    return tri_inv_trick_fp16


def build_kernel(matrix_size: int, *, double_buffer: bool = False):
    if double_buffer:
        return build_double_buffer_kernel(matrix_size)
    return build_single_buffer_kernel(matrix_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-size",
        type=int,
        choices=SUPPORTED_MATRIX_SIZES,
        default=64,
        help="Compile-time specialized dense matrix size.",
    )
    parser.add_argument(
        "--double-buffer",
        action="store_true",
        help="Enable L1/L0 ping-pong buffers for the iterative recurrence.",
    )
    args = parser.parse_args()
    module = build_kernel(args.matrix_size, double_buffer=args.double_buffer)
    print(module)
