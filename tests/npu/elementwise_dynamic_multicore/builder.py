from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const


DTYPES = {
    "float32": lambda: pto.float32,
    "float16": lambda: pto.float16,
    "int32": lambda: pto.int32,
    "int16": lambda: pto.int16,
}


def meta_data(dtype=None, tile_length=1024):
    if dtype is None:
        dtype = "float32"
    if isinstance(dtype, str):
        dtype = DTYPES[dtype]()
    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=1, dtype=dtype)
    subtensor_type = pto.SubTensorType(shape=[1, tile_length], dtype=dtype)
    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[1, tile_length],
        valid_shape=[1, tile_length],
        dtype=dtype,
        memory_space="VEC",
        config=tile_cfg,
    )
    return {
        "ptr_type": ptr_type,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_type": subtensor_type,
        "tile_type": tile_type,
        "tile_length": tile_length,
    }


def build_binary_kernels(op_name, op_fn, dtype=None, tile_length=1024):
    """Build 1D and 2D dynamic multicore kernels for a binary elementwise op.
    op_name: name of the op, used in generated function names and file names.
    op_fn: function that takes two tile buffers and produces the result in a third tile buffer
    dtype: element type (default float32). Use "int16" for bitwise ops.
    tile_length: tile width (default 1024).
    Returns (kernel_1d_module, kernel_2d_module).
    """
    _meta_data = lambda: meta_data(dtype, tile_length)

    def _1d(
        arg0: "ptr_type", arg1: "ptr_type", arg2: "ptr_type", argN: "index_dtype"
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(tile_length)

        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        cidmul = cid * sub_bnum
        vid = cidmul + sub_bid
        num_blocks = pto.get_block_num()

        vid_idx = pto.index_cast(vid)
        num_cores = pto.index_cast(num_blocks)
        total_elements = pto.index_cast(argN)

        num_tiles_global = pto.ceil_div(total_elements, c_tile)
        num_tiles_per_core = pto.ceil_div(num_tiles_global, num_cores)
        tile_offset_this_core = vid_idx * num_tiles_per_core

        with pto.vector_section():
            tv0 = pto.as_tensor(
                tensor_type, ptr=arg0, shape=[total_elements], strides=[c1]
            )
            tv1 = pto.as_tensor(
                tensor_type, ptr=arg1, shape=[total_elements], strides=[c1]
            )
            tv2 = pto.as_tensor(
                tensor_type, ptr=arg2, shape=[total_elements], strides=[c1]
            )

            tb0 = pto.alloc_tile(tile_type)
            tb1 = pto.alloc_tile(tile_type)
            tb2 = pto.alloc_tile(tile_type)

            with pto.if_(tile_offset_this_core < num_tiles_global):
                tiles_end_this_core = tile_offset_this_core + num_tiles_per_core
                need_truncate = tiles_end_this_core > num_tiles_global
                remaining_tiles = num_tiles_global - tile_offset_this_core
                tiles_to_process = pto.select(
                    need_truncate, remaining_tiles, num_tiles_per_core
                )
                elements_to_process = tiles_to_process * c_tile

                with pto.if_(elements_to_process > c0):
                    for i in pto.for_range(c0, tiles_to_process, c1):
                        tile_offset_global = i + tile_offset_this_core
                        offset_global = tile_offset_global * c_tile

                        sv0 = pto.slice_view(
                            subtensor_type,
                            source=tv0,
                            offsets=[offset_global],
                            sizes=[c_tile],
                        )
                        sv1 = pto.slice_view(
                            subtensor_type,
                            source=tv1,
                            offsets=[offset_global],
                            sizes=[c_tile],
                        )
                        sv2 = pto.slice_view(
                            subtensor_type,
                            source=tv2,
                            offsets=[offset_global],
                            sizes=[c_tile],
                        )

                        pto.load(sv0, tb0)
                        pto.load(sv1, tb1)
                        op_fn(tb0, tb1, tb2)
                        pto.store(tb2, sv2)

    _1d.__name__ = f"vec_{op_name}_1d_dynamic"
    kernel_1d = to_ir_module(meta_data=_meta_data)(_1d)

    def _2d(
        arg0: "ptr_type",
        arg1: "ptr_type",
        arg2: "ptr_type",
        argM: "index_dtype",
        argN: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c_tile = const(tile_length)

        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        cidmul = cid * sub_bnum
        vid = cidmul + sub_bid
        num_blocks = pto.get_block_num()

        vid_idx = pto.index_cast(vid)
        num_cores = pto.index_cast(num_blocks)
        rows = pto.index_cast(argM)
        cols = pto.index_cast(argN)

        total_elements = rows * cols
        rows_per_core = pto.ceil_div(rows, num_cores)
        row_start = vid_idx * rows_per_core
        tiles_per_row = pto.ceil_div(cols, c_tile)

        with pto.vector_section():
            tv0 = pto.as_tensor(
                tensor_type, ptr=arg0, shape=[total_elements], strides=[c1]
            )
            tv1 = pto.as_tensor(
                tensor_type, ptr=arg1, shape=[total_elements], strides=[c1]
            )
            tv2 = pto.as_tensor(
                tensor_type, ptr=arg2, shape=[total_elements], strides=[c1]
            )

            tb0 = pto.alloc_tile(tile_type)
            tb1 = pto.alloc_tile(tile_type)
            tb2 = pto.alloc_tile(tile_type)

            with pto.if_(row_start < rows):
                rows_end = row_start + rows_per_core
                need_truncate = rows_end > rows
                remaining_rows = rows - row_start
                rows_to_process = pto.select(
                    need_truncate, remaining_rows, rows_per_core
                )

                for r in pto.for_range(c0, rows_to_process, c1):
                    row_idx = r + row_start
                    row_flat_offset = row_idx * cols
                    for c in pto.for_range(c0, tiles_per_row, c1):
                        col_offset = c * c_tile
                        flat_offset = row_flat_offset + col_offset

                        sv0 = pto.slice_view(
                            subtensor_type,
                            source=tv0,
                            offsets=[flat_offset],
                            sizes=[c_tile],
                        )
                        sv1 = pto.slice_view(
                            subtensor_type,
                            source=tv1,
                            offsets=[flat_offset],
                            sizes=[c_tile],
                        )
                        sv2 = pto.slice_view(
                            subtensor_type,
                            source=tv2,
                            offsets=[flat_offset],
                            sizes=[c_tile],
                        )

                        pto.load(sv0, tb0)
                        pto.load(sv1, tb1)
                        op_fn(tb0, tb1, tb2)
                        pto.store(tb2, sv2)

    _2d.__name__ = f"vec_{op_name}_2d_dynamic"
    kernel_2d = to_ir_module(meta_data=_meta_data)(_2d)

    return kernel_1d, kernel_2d
