from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

_TILE_ROWS = 32
_TILE_COLS = 32


def build_sym_dynamic():
    """Dynamic multicore symmetric quantization kernel.

    Args:
        src_ptr   : float32[batch, n_cols]  input
        fp_ptr    : float32[batch, n_cols]  per-element scale factors
        dst_ptr   : int8[batch, n_cols]     quantized output
        batch     : int32
        n_cols    : int32  (must be a multiple of _TILE_COLS = 32)

    Semantics:
        dst[i, j] = int8(round(src[i, j] * fp[i, j]))
    """

    def _meta():
        return {
            "ptr_f32": pto.PtrType(pto.float32),
            "ptr_i8": pto.PtrType(pto.int8),
            "index_dtype": pto.int32,
            "tensor_f32": pto.TensorType(rank=2, dtype=pto.float32),
            "tensor_i8": pto.TensorType(rank=2, dtype=pto.int8),
            "sub_f32": pto.SubTensorType(
                shape=[_TILE_ROWS, _TILE_COLS], dtype=pto.float32
            ),
            "sub_i8": pto.SubTensorType(shape=[_TILE_ROWS, _TILE_COLS], dtype=pto.int8),
            "tile_f32": pto.TileBufType(
                shape=[_TILE_ROWS, _TILE_COLS],
                valid_shape=[-1, -1],
                dtype=pto.float32,
                memory_space="VEC",
                config=pto.TileBufConfig(),
            ),
            "tile_i8": pto.TileBufType(
                shape=[_TILE_ROWS, _TILE_COLS],
                valid_shape=[-1, -1],
                dtype=pto.int8,
                memory_space="VEC",
                config=pto.TileBufConfig(),
            ),
        }

    @to_ir_module(meta_data=_meta)
    def quant_sym_dynamic(
        src_ptr: "ptr_f32",
        fp_ptr: "ptr_f32",
        dst_ptr: "ptr_i8",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile_rows = const(_TILE_ROWS)
        c_tile_cols = const(_TILE_COLS)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            num_cores = s.index_cast(pto.get_block_num())

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, batch)

            tv_src = pto.as_tensor(
                tensor_f32, ptr=src_ptr, shape=[batch, n_cols], strides=[n_cols, c1]
            )
            tv_fp = pto.as_tensor(
                tensor_f32, ptr=fp_ptr, shape=[batch, n_cols], strides=[n_cols, c1]
            )
            tv_dst = pto.as_tensor(
                tensor_i8, ptr=dst_ptr, shape=[batch, n_cols], strides=[n_cols, c1]
            )

            for row in pto.range(row_start, row_end, c_tile_rows):
                rows_this = s.min_u(c_tile_rows, row_end - row)

                for col in pto.range(c0, n_cols, c_tile_cols):
                    tb_src = pto.alloc_tile(
                        tile_f32, valid_row=rows_this, valid_col=c_tile_cols
                    )
                    tb_fp = pto.alloc_tile(
                        tile_f32, valid_row=rows_this, valid_col=c_tile_cols
                    )
                    tb_dst = pto.alloc_tile(
                        tile_i8, valid_row=rows_this, valid_col=c_tile_cols
                    )

                    sv_src = pto.slice_view(
                        sub_f32,
                        source=tv_src,
                        offsets=[row, col],
                        sizes=[rows_this, c_tile_cols],
                    )
                    sv_fp = pto.slice_view(
                        sub_f32,
                        source=tv_fp,
                        offsets=[row, col],
                        sizes=[rows_this, c_tile_cols],
                    )
                    sv_dst = pto.slice_view(
                        sub_i8,
                        source=tv_dst,
                        offsets=[row, col],
                        sizes=[rows_this, c_tile_cols],
                    )

                    pto.load(sv_src, tb_src)
                    pto.load(sv_fp, tb_fp)
                    tile.quant(tb_src, tb_fp, tb_dst, "int8_sym")
                    pto.store(tb_dst, sv_dst)

    return quant_sym_dynamic


def build_asym_dynamic():
    """Dynamic multicore asymmetric quantization kernel.

    Args:
        src_ptr    : float32[batch, n_cols]  input
        fp_ptr     : float32[batch, n_cols]  per-element scale factors
        offset_ptr : float32[batch, n_cols]  per-element offsets
        dst_ptr    : uint8[batch, n_cols]    quantized output
        batch      : int32
        n_cols     : int32  (must be a multiple of _TILE_COLS = 32)

    Semantics:
        dst[i, j] = uint8(round(src[i, j] * fp[i, j]) + offset[i, j])
    """

    def _meta():
        return {
            "ptr_f32": pto.PtrType(pto.float32),
            "ptr_u8": pto.PtrType(pto.uint8),
            "index_dtype": pto.int32,
            "tensor_f32": pto.TensorType(rank=2, dtype=pto.float32),
            "tensor_u8": pto.TensorType(rank=2, dtype=pto.uint8),
            "sub_f32": pto.SubTensorType(
                shape=[_TILE_ROWS, _TILE_COLS], dtype=pto.float32
            ),
            "sub_u8": pto.SubTensorType(
                shape=[_TILE_ROWS, _TILE_COLS], dtype=pto.uint8
            ),
            "tile_f32": pto.TileBufType(
                shape=[_TILE_ROWS, _TILE_COLS],
                valid_shape=[-1, -1],
                dtype=pto.float32,
                memory_space="VEC",
                config=pto.TileBufConfig(),
            ),
            "tile_u8": pto.TileBufType(
                shape=[_TILE_ROWS, _TILE_COLS],
                valid_shape=[-1, -1],
                dtype=pto.uint8,
                memory_space="VEC",
                config=pto.TileBufConfig(),
            ),
        }

    @to_ir_module(meta_data=_meta)
    def quant_asym_dynamic(
        src_ptr: "ptr_f32",
        fp_ptr: "ptr_f32",
        offset_ptr: "ptr_f32",
        dst_ptr: "ptr_u8",
        batch_i32: "index_dtype",
        n_cols_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile_rows = const(_TILE_ROWS)
        c_tile_cols = const(_TILE_COLS)

        batch = s.index_cast(batch_i32)
        n_cols = s.index_cast(n_cols_i32)

        with pto.vector_section():
            bid = s.index_cast(pto.get_block_idx())
            num_cores = s.index_cast(pto.get_block_num())

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = bid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, batch)

            tv_src = pto.as_tensor(
                tensor_f32, ptr=src_ptr, shape=[batch, n_cols], strides=[n_cols, c1]
            )
            tv_fp = pto.as_tensor(
                tensor_f32, ptr=fp_ptr, shape=[batch, n_cols], strides=[n_cols, c1]
            )
            tv_offset = pto.as_tensor(
                tensor_f32, ptr=offset_ptr, shape=[batch, n_cols], strides=[n_cols, c1]
            )
            tv_dst = pto.as_tensor(
                tensor_u8, ptr=dst_ptr, shape=[batch, n_cols], strides=[n_cols, c1]
            )

            for row in pto.range(row_start, row_end, c_tile_rows):
                rows_this = s.min_u(c_tile_rows, row_end - row)

                for col in pto.range(c0, n_cols, c_tile_cols):
                    tb_src = pto.alloc_tile(
                        tile_f32, valid_row=rows_this, valid_col=c_tile_cols
                    )
                    tb_fp = pto.alloc_tile(
                        tile_f32, valid_row=rows_this, valid_col=c_tile_cols
                    )
                    tb_offset = pto.alloc_tile(
                        tile_f32, valid_row=rows_this, valid_col=c_tile_cols
                    )
                    tb_dst = pto.alloc_tile(
                        tile_u8, valid_row=rows_this, valid_col=c_tile_cols
                    )

                    sv_src = pto.slice_view(
                        sub_f32,
                        source=tv_src,
                        offsets=[row, col],
                        sizes=[rows_this, c_tile_cols],
                    )
                    sv_fp = pto.slice_view(
                        sub_f32,
                        source=tv_fp,
                        offsets=[row, col],
                        sizes=[rows_this, c_tile_cols],
                    )
                    sv_offset = pto.slice_view(
                        sub_f32,
                        source=tv_offset,
                        offsets=[row, col],
                        sizes=[rows_this, c_tile_cols],
                    )
                    sv_dst = pto.slice_view(
                        sub_u8,
                        source=tv_dst,
                        offsets=[row, col],
                        sizes=[rows_this, c_tile_cols],
                    )

                    pto.load(sv_src, tb_src)
                    pto.load(sv_fp, tb_fp)
                    pto.load(sv_offset, tb_offset)
                    tile.quant(tb_src, tb_fp, tb_dst, "int8_asym", offset=tb_offset)
                    pto.store(tb_dst, sv_dst)

    return quant_asym_dynamic


if __name__ == "__main__":
    print(build_sym_dynamic())
