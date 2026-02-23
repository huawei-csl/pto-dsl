from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const

_DTYPE_MAP = {
    "float32": lambda: pto.float32,
    "float16": lambda: pto.float16,
    "int16": lambda: pto.int16,
    "int32": lambda: pto.int32,
}


def meta_data(src_dtype=None, rows=32, cols=32):
    if src_dtype is None:
        src_dtype = pto.float32
    if isinstance(src_dtype, str):
        src_dtype = _DTYPE_MAP[src_dtype]()

    i32 = pto.int32
    tile_cfg = pto.TileBufConfig()

    return {
        "ptr_src": pto.PtrType(src_dtype),
        "ptr_i32": pto.PtrType(i32),
        "tv2_src": pto.TensorType(rank=2, dtype=src_dtype),
        "tv2_i32": pto.TensorType(rank=2, dtype=i32),
        "tile_view_src": pto.SubTensorType(shape=[rows, cols], dtype=src_dtype),
        "tile_view_i32": pto.SubTensorType(shape=[rows, cols], dtype=i32),
        "tile_buf_src": pto.TileBufType(
            shape=[rows, cols],
            valid_shape=[rows, cols],
            dtype=src_dtype,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_buf_i32": pto.TileBufType(
            shape=[rows, cols],
            valid_shape=[rows, cols],
            dtype=i32,
            memory_space="VEC",
            config=tile_cfg,
        ),
    }


def build_gather_kernel(
    fn_name="vec_gather_kernel", dtype=None, rows=32, cols=32, mask_pattern="P1111"
):
    _meta_data = lambda: meta_data(dtype, rows, cols)
    assert mask_pattern in (
        "P1111",
        "P0101",
        "P0001",
    ), f"Unsupported mask_pattern: {mask_pattern}"

    def _kernel(
        arg0: "ptr_src",  # src   (T*)
        arg1: "ptr_i32",  # indices (i32*)
        arg2: "ptr_src",  # dst   (T*)
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        crow = const(rows)
        ccol = const(cols)

        with pto.vector_section():
            tv0 = pto.as_tensor(
                tv2_src, ptr=arg0, shape=[crow, ccol], strides=[ccol, c1]
            )
            tv1 = pto.as_tensor(
                tv2_i32, ptr=arg1, shape=[crow, ccol], strides=[ccol, c1]
            )
            tv2 = pto.as_tensor(
                tv2_src, ptr=arg2, shape=[crow, ccol], strides=[ccol, c1]
            )

            sv0 = pto.slice_view(
                tile_view_src, source=tv0, offsets=[c0, c0], sizes=[crow, ccol]
            )
            sv1 = pto.slice_view(
                tile_view_i32, source=tv1, offsets=[c0, c0], sizes=[crow, ccol]
            )

            tb0 = pto.alloc_tile(tile_buf_src)
            tb1 = pto.alloc_tile(tile_buf_i32)
            tb2 = pto.alloc_tile(tile_buf_src)  # tmp
            tb3 = pto.alloc_tile(tile_buf_src)  # out

            pto.load(sv0, tb0)
            pto.load(sv1, tb1)

            pto.gather(tb0, tb2, tb1)  # index-gather: tb2[i,j] = tb0[tb1[i,j]]
            pto.gather(
                tb2, tb3, mask_pattern=mask_pattern
            )  # mask-gather with configurable pattern

            sv2 = pto.slice_view(
                tile_view_src, source=tv2, offsets=[c0, c0], sizes=[crow, ccol]
            )
            pto.store(tb3, sv2)

    _kernel.__name__ = fn_name
    return to_ir_module(meta_data=_meta_data)(_kernel)


if __name__ == "__main__":
    print(build_gather_kernel(mask_pattern="P1111"))
