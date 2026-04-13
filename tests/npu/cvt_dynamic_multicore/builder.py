from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

# Tile dimensions.
_TILE_ROWS = 32
_TILE_COLS = 32

_SUPPORTED_DTYPES = frozenset(
    ["float32", "float16", "int64", "int32", "int16", "int8", "uint8"]
)


def _pto_type(name):
    """Resolve a dtype string to a PTO type (must be called inside an MLIR context)."""
    return getattr(pto, name)


def build_cvt(src_dtype, dst_dtype, rmode=None):
    """Generic dynamic multicore type conversion kernel.

    Args:
        src_ptr  : src_dtype[batch, n_cols]  input
        dst_ptr  : dst_dtype[batch, n_cols]  output
        batch    : int32
        n_cols   : int32  (must be a multiple of _TILE_COLS = 32)
        rmode    : optional rounding mode string passed to tile.cvt

    Semantics:
        dst[i, j] = dst_dtype(src[i, j])
    """

    def _meta():
        # PTO types must be resolved here, inside the MLIR context set up by to_ir_module.
        src_pto = _pto_type(src_dtype)
        dst_pto = _pto_type(dst_dtype)
        return {
            "ptr_src": pto.PtrType(src_pto),
            "ptr_dst": pto.PtrType(dst_pto),
            "index_dtype": pto.int32,
            "tensor_src": pto.TensorType(rank=2, dtype=src_pto),
            "tensor_dst": pto.TensorType(rank=2, dtype=dst_pto),
            "sub_src": pto.SubTensorType(shape=[_TILE_ROWS, _TILE_COLS], dtype=src_pto),
            "sub_dst": pto.SubTensorType(shape=[_TILE_ROWS, _TILE_COLS], dtype=dst_pto),
            "tile_src": pto.TileBufType(
                shape=[_TILE_ROWS, _TILE_COLS],
                valid_shape=[-1, -1],
                dtype=src_pto,
                memory_space="VEC",
                config=pto.TileBufConfig(),
            ),
            "tile_dst": pto.TileBufType(
                shape=[_TILE_ROWS, _TILE_COLS],
                valid_shape=[-1, -1],
                dtype=dst_pto,
                memory_space="VEC",
                config=pto.TileBufConfig(),
            ),
        }

    @to_ir_module(meta_data=_meta)
    def _kernel(
        src_ptr: "ptr_src",
        dst_ptr: "ptr_dst",
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
            cid = pto.get_block_idx()
            sub_bid = pto.get_subblock_idx()
            sub_bnum = pto.get_subblock_num()
            vid = s.index_cast(cid * sub_bnum + sub_bid)
            num_cores = s.index_cast(pto.get_block_num() * sub_bnum)

            rows_per_core = s.ceil_div(batch, num_cores)
            row_start = vid * rows_per_core
            row_end = s.min_u(row_start + rows_per_core, batch)

            tv_src = pto.as_tensor(
                tensor_src, ptr=src_ptr, shape=[batch, n_cols], strides=[n_cols, c1]
            )
            tv_dst = pto.as_tensor(
                tensor_dst, ptr=dst_ptr, shape=[batch, n_cols], strides=[n_cols, c1]
            )

            for row in pto.range(row_start, row_end, c_tile_rows):
                rows_this = s.min_u(c_tile_rows, row_end - row)

                for col in pto.range(c0, n_cols, c_tile_cols):
                    tb_src = pto.alloc_tile(
                        tile_src, valid_row=rows_this, valid_col=c_tile_cols
                    )
                    tb_dst = pto.alloc_tile(
                        tile_dst, valid_row=rows_this, valid_col=c_tile_cols
                    )

                    sv_src = pto.slice_view(
                        sub_src,
                        source=tv_src,
                        offsets=[row, col],
                        sizes=[rows_this, c_tile_cols],
                    )
                    sv_dst = pto.slice_view(
                        sub_dst,
                        source=tv_dst,
                        offsets=[row, col],
                        sizes=[rows_this, c_tile_cols],
                    )

                    pto.load(sv_src, tb_src)
                    tile.cvt(tb_src, tb_dst, rmode=rmode)
                    pto.store(tb_dst, sv_dst)

    return _kernel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dtype", required=True, choices=sorted(_SUPPORTED_DTYPES))
    parser.add_argument("--dst-dtype", required=True, choices=sorted(_SUPPORTED_DTYPES))
    parser.add_argument("--rmode", default=None)
    args = parser.parse_args()

    print(build_cvt(args.src_dtype, args.dst_dtype, rmode=args.rmode))
