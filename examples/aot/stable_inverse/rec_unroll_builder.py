# pyright: reportUndefinedVariable=false
import argparse

from mlir.dialects import arith
from mlir.ir import IntegerAttr, IntegerType

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const
SUPPORTED_MATRIX_SIZES = (16, 32, 64, 128)


def i64_const(value: int):
    i64 = IntegerType.get_signless(64)
    return arith.ConstantOp(i64, IntegerAttr.get(i64, value)).result


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
        l0a_fractal_type = pto.TileBufType(
            shape=[16, 16],
            valid_shape=[16, 16],
            dtype=in_dtype,
            memory_space="LEFT",
        )
        l0b_fractal_type = pto.TileBufType(
            shape=[16, 16],
            valid_shape=[16, 16],
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
            "l0a_fractal_type": l0a_fractal_type,
            "l0b_fractal_type": l0b_fractal_type,
            "l0c_tile_type": l0c_tile_type,
        }

    return meta_data


def _copy_diagonal_fractals_to_l0(
    src_l1, fractal_type, matrix_size: int, dst_base_addr_bytes: int
):
    num_fractals = matrix_size // 16
    elem_bytes = 2  # fp16
    for i in range(num_fractals):
        rc = i * 16
        offset_elems = i * 16 * (matrix_size + 16)
        dst_fractal = pto.alloc_tile(
            fractal_type, addr=i64_const(dst_base_addr_bytes + offset_elems * elem_bytes)
        )
        tile.extract(src_l1, const(rc), const(rc), dst_fractal)


def _copy_odd_even_blocks_to_l0(
    src_l1,
    fractal_type,
    matrix_size: int,
    block_size: int,
    start_block_index: int,
    dst_base_addr_bytes: int,
):
    num_blocks = matrix_size // block_size
    num_fractals_per_block = block_size // 16
    elem_bytes = 2  # fp16
    for i in range(num_fractals_per_block):
        for j in range(num_fractals_per_block):
            for b in range(start_block_index, num_blocks, 2):
                row = b * block_size + i * 16
                col = b * block_size + j * 16
                offset_elems = (
                    b * (matrix_size + 16) * block_size
                    + i * matrix_size * 16
                    + j * 16 * 16
                )
                dst_fractal = pto.alloc_tile(
                    fractal_type,
                    addr=i64_const(dst_base_addr_bytes + offset_elems * elem_bytes),
                )
                tile.extract(src_l1, const(row), const(col), dst_fractal)


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
            c2 = const(2)
            c4 = const(4)
            matrix_size_c = const(matrix_size)
            matrix_size_sq = const(matrix_size * matrix_size)
            fractal_half = const(8)  # FractalSize / 2, FractalSize fixed at 16.
            fractal_quarter = const(4)  # FractalSize / 4.
            tile_len = matrix_size * matrix_size
            bytes_f16_tile = tile_len * 2
            bytes_f32_tile = tile_len * 4

            # Keep ABI identical to the C++ kernel. matrix_size is specialized by builder.
            _ = matrix_size_i32
            _ = num_bsnd_heads_i32
            _ = manual_sync  # TODO: add explicit manual sync events for this kernel.

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

            # Explicit addresses are required for --pto-level=level3.
            i_l1_base = 0 * bytes_f16_tile
            i_neg_l1_base = 1 * bytes_f16_tile
            zero_l1_base = 2 * bytes_f16_tile
            m_neg_l1_base = 3 * bytes_f16_tile
            x_l1_base = 4 * bytes_f16_tile
            y_l1_base = 5 * bytes_f16_tile

            a0_base = 0 * bytes_f16_tile
            a1_base = 1 * bytes_f16_tile
            b0_base = 0 * bytes_f16_tile
            b1_base = 1 * bytes_f16_tile
            c0_base = 0 * bytes_f32_tile
            c1_base = 1 * bytes_f32_tile

            i_l1 = pto.alloc_tile(l1_tile_type, addr=i64_const(i_l1_base))
            i_neg_l1 = pto.alloc_tile(l1_tile_type, addr=i64_const(i_neg_l1_base))
            zero_l1 = pto.alloc_tile(l1_tile_type, addr=i64_const(zero_l1_base))
            m_neg_l1 = pto.alloc_tile(l1_tile_type, addr=i64_const(m_neg_l1_base))
            x_l1 = pto.alloc_tile(l1_tile_type, addr=i64_const(x_l1_base))
            y_l1 = pto.alloc_tile(l1_tile_type, addr=i64_const(y_l1_base))

            a_l0 = [
                pto.alloc_tile(l0a_tile_type, addr=i64_const(a0_base)),
                pto.alloc_tile(l0a_tile_type, addr=i64_const(a1_base)),
            ]
            b_l0 = [
                pto.alloc_tile(l0b_tile_type, addr=i64_const(b0_base)),
                pto.alloc_tile(l0b_tile_type, addr=i64_const(b1_base)),
            ]
            c_l0 = [
                pto.alloc_tile(l0c_tile_type, addr=i64_const(c0_base)),
                pto.alloc_tile(l0c_tile_type, addr=i64_const(c1_base)),
            ]
            sv_i_neg = pto.slice_view(
                i_neg_subtensor,
                source=tv_i_neg,
                offsets=[c0, c0],
                sizes=[matrix_size_c, matrix_size_c],
            )
            pto.load(sv_i_neg, i_neg_l1)
            # Prepare identity and zero in L1.
            tile.mov(i_neg_l1, a_l0[0])
            tile.mov(i_neg_l1, b_l0[0])
            tile.matmul(a_l0[0], b_l0[0], c_l0[0])  # I
            tile.mov(c_l0[0], i_l1)
            tile.mov(i_l1, b_l0[0])
            tile.matmul_acc(c_l0[0], a_l0[0], b_l0[0], c_l0[0])  # 0
            tile.mov(c_l0[0], zero_l1)

            # Build-time unrolled block sizes: 16,32,64,...,<matrix_size
            recursion_blocks = []
            bs = 16
            while bs < matrix_size:
                recursion_blocks.append(bs)
                bs *= 2

            final_c_idx = 1 if matrix_size > 16 else 0
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
                # M_neg <- (-I) @ M
                tile.mov(y_l1, b_l0[0])
                tile.mov(i_neg_l1, a_l0[0])
                tile.matmul(a_l0[0], b_l0[0], c_l0[0])
                tile.mov(c_l0[0], m_neg_l1)

                # a_l0[1], b_l0[1] <- diagonal fractals of M
                tile.mov(zero_l1, a_l0[1])
                tile.mov(zero_l1, b_l0[1])
                _copy_diagonal_fractals_to_l0(
                    y_l1, l0a_fractal_type, matrix_size, dst_base_addr_bytes=a1_base
                )
                _copy_diagonal_fractals_to_l0(
                    y_l1, l0b_fractal_type, matrix_size, dst_base_addr_bytes=b1_base
                )

                # Y <- diag(M) @ diag(M)
                tile.matmul(a_l0[1], b_l0[1], c_l0[1])
                tile.mov(c_l0[1], y_l1)

                # X <- I - diag(M)
                tile.mov(i_neg_l1, b_l0[0])
                tile.mov(i_neg_l1, a_l0[0])
                tile.matmul(a_l0[1], b_l0[0], c_l0[0])  # diag(M_neg)
                tile.matmul_acc(c_l0[0], a_l0[0], b_l0[0], c_l0[0])  # + I
                tile.mov(c_l0[0], x_l1)

                # Inv-trick phase over fractal blocks (FractalSize=16).
                for loop_i in (c1, c2, c4):
                    tile.mov(i_l1, b_l0[0])
                    tile.mov(x_l1, a_l0[0])
                    tile.matmul(a_l0[0], b_l0[0], c_l0[0])  # X

                    tile.mov(y_l1, b_l0[1])
                    with pto.if_context(loop_i < fractal_quarter):
                        tile.mov(y_l1, a_l0[1])
                        tile.matmul(a_l0[1], b_l0[1], c_l0[1])  # Y^2
                        tile.mov(c_l0[1], y_l1)

                    tile.matmul_acc(c_l0[0], a_l0[0], b_l0[1], c_l0[0])  # X + X@Y
                    with pto.if_context(loop_i < fractal_half):
                        tile.mov(c_l0[0], x_l1)

                # Unrolled recursion phase.
                tile.mov(m_neg_l1, b_l0[1])  # fixed rhs for LX @ (-M)
                tile.mov(i_l1, a_l0[0])  # fixed lhs for I
                for block_size in recursion_blocks:
                    tile.mov(zero_l1, a_l0[1])
                    _copy_odd_even_blocks_to_l0(
                        x_l1,
                        l0a_fractal_type,
                        matrix_size,
                        block_size,
                        start_block_index=0,
                        dst_base_addr_bytes=a1_base,
                    )  # LX
                    tile.mov(i_l1, b_l0[0])

                    tile.matmul(a_l0[0], b_l0[0], c_l0[0])  # I
                    tile.matmul(a_l0[1], b_l0[0], c_l0[1])  # LX
                    tile.matmul_acc(c_l0[0], a_l0[1], b_l0[1], c_l0[0])  # Y
                    tile.mov(c_l0[0], y_l1)

                    tile.mov(zero_l1, b_l0[0])
                    _copy_odd_even_blocks_to_l0(
                        x_l1,
                        l0b_fractal_type,
                        matrix_size,
                        block_size,
                        start_block_index=1,
                        dst_base_addr_bytes=b0_base,
                    )  # RX
                    tile.mov(y_l1, a_l0[1])
                    tile.matmul_acc(c_l0[1], a_l0[1], b_l0[0], c_l0[1])  # Y@RX + LX
                    if block_size < (matrix_size // 2):
                        tile.mov(c_l0[1], x_l1)

                pto.store(c_l0[final_c_idx], sv_out)

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
