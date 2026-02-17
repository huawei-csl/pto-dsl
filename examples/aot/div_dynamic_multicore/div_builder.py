from ptodsl import to_ir_module
import ptodsl.language as pto
const = pto.const


def meta_data():
    # common, reusable type declarations
    dtype = pto.float32
    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=1, dtype=dtype)
    tile_length = 1024
    subtensor_type = pto.SubTensorType(shape=[1, tile_length], dtype=dtype)
    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[1, tile_length], valid_shape=[1, tile_length], dtype=dtype, memory_space="VEC", config=tile_cfg)
    return {
        "ptr_type": ptr_type,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_type": subtensor_type,
        "tile_type": tile_type,
        "tile_length": tile_length,
    }


@to_ir_module(meta_data=meta_data)
def vec_div_1d_dynamic(
    arg0: "ptr_type",
    arg1: "ptr_type",
    arg2: "ptr_type",
    argN: "index_dtype"
    ) -> None:
    c0 = const(0)
    c1 = const(1)
    c_tile = const(tile_length)

    # Get core/block information
    cid = pto.get_block_idx()
    sub_bid = pto.get_subblock_idx()
    sub_bnum = pto.get_subblock_num()
    cidmul = cid * sub_bnum
    vid = cidmul + sub_bid
    num_blocks = pto.get_block_num()

    # Convert to index types
    vid_idx = pto.index_cast(vid)
    num_cores = pto.index_cast(num_blocks)
    total_elements = pto.index_cast(argN)

    # Distribute tiles across cores using ceil division
    num_tiles_global = pto.ceil_div(total_elements, c_tile)
    num_tiles_per_core = pto.ceil_div(num_tiles_global, num_cores)
    tile_offset_this_core = vid_idx * num_tiles_per_core

    with pto.vector_section():
        # Create tensor views
        tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[total_elements], strides=[c1])
        tv1 = pto.as_tensor(tensor_type, ptr=arg1, shape=[total_elements], strides=[c1])
        tv2 = pto.as_tensor(tensor_type, ptr=arg2, shape=[total_elements], strides=[c1])

        # Allocate tiles once
        tb0 = pto.alloc_tile(tile_type)
        tb1 = pto.alloc_tile(tile_type)
        tb2 = pto.alloc_tile(tile_type)

        # Skip whole core if its starting tile is already out-of-bound.
        with pto.if_(pto.lt(tile_offset_this_core, num_tiles_global)):
            tiles_end_this_core = tile_offset_this_core + num_tiles_per_core
            need_truncate = pto.gt(tiles_end_this_core, num_tiles_global)
            remaining_tiles = num_tiles_global - tile_offset_this_core

            # Truncate per-core tiles when the chunk crosses global bound.
            tiles_to_process = pto.select(need_truncate, remaining_tiles, num_tiles_per_core)
            elements_to_process = tiles_to_process * c_tile

            with pto.if_(pto.gt(elements_to_process, c0)):
                for i in pto.for_range(c0, tiles_to_process, c1):
                    tile_offset_global = i + tile_offset_this_core
                    offset_global = tile_offset_global * c_tile

                    sv0 = pto.slice_view(subtensor_type, source=tv0, offsets=[offset_global], sizes=[c_tile])
                    sv1 = pto.slice_view(subtensor_type, source=tv1, offsets=[offset_global], sizes=[c_tile])
                    sv2 = pto.slice_view(subtensor_type, source=tv2, offsets=[offset_global], sizes=[c_tile])

                    pto.load(sv0, tb0)
                    pto.load(sv1, tb1)
                    pto.div(tb0, tb1, tb2)
                    pto.store(tb2, sv2)


if __name__ == "__main__":
    module = vec_div_1d_dynamic
    print(module)
