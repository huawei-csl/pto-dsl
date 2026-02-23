from ptodsl import jit
import ptodsl.language as pto
import torch
import torch_npu

const = pto.const


def meta_data():
    dtype = pto.float32
    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=1, dtype=dtype)
    tile_length = 1024  # TODO: increase to 8192 for better DMA util
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


@jit(meta_data=meta_data, block_dim=20)
def vec_add_1d_dynamic(
    arg0: "ptr_type",
    arg1: "ptr_type",
    arg2: "ptr_type",
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

    # Convert i64/i32 values to index for arithmetic ops.
    vid_idx = pto.index_cast(vid)
    num_cores = pto.index_cast(num_blocks)
    total_elements = pto.index_cast(argN)

    num_tiles_global = pto.ceil_div(total_elements, c_tile)
    num_tiles_per_core = pto.ceil_div(num_tiles_global, num_cores)
    tile_offset_this_core = vid_idx * num_tiles_per_core

    with pto.vector_section():
        tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[total_elements], strides=[c1])
        tv1 = pto.as_tensor(tensor_type, ptr=arg1, shape=[total_elements], strides=[c1])
        tv2 = pto.as_tensor(tensor_type, ptr=arg2, shape=[total_elements], strides=[c1])

        tb0 = pto.alloc_tile(tile_type)
        tb1 = pto.alloc_tile(tile_type)
        tb2 = pto.alloc_tile(tile_type)

        # Skip whole core if its starting tile is already out-of-bound.
        with pto.if_(tile_offset_this_core < num_tiles_global):
            tiles_end_this_core = tile_offset_this_core + num_tiles_per_core
            need_truncate = tiles_end_this_core > num_tiles_global
            remaining_tiles = num_tiles_global - tile_offset_this_core

            # Keep the same scf.if with result pattern as the verbose IR builder.
            tiles_to_process = pto.if_else_yield(
                need_truncate, remaining_tiles, num_tiles_per_core
            )

            elements_to_process = tiles_to_process * c_tile
            with pto.if_(elements_to_process > c0):
                for i in pto.for_range(c0, tiles_to_process, c1):
                    tile_offset_global = i + tile_offset_this_core
                    offset_global = tile_offset_global * c_tile

                    sv0 = pto.slice_view(
                        subtensor_type, source=tv0, offsets=[offset_global], sizes=[c_tile]
                    )
                    sv1 = pto.slice_view(
                        subtensor_type, source=tv1, offsets=[offset_global], sizes=[c_tile]
                    )
                    sv2 = pto.slice_view(
                        subtensor_type, source=tv2, offsets=[offset_global], sizes=[c_tile]
                    )

                    pto.load(sv0, tb0)
                    pto.load(sv1, tb1)
                    pto.add(tb0, tb1, tb2)
                    pto.store(tb2, sv2)


def test_add():
    device = "npu:7"
    torch.npu.set_device(device)

    # shape parameter hard-coded as kernel
    num_cores = 20 * 2
    tile_size = 1024
    # Keep shapes aligned to tile size, but vary tile counts so they are not
    # required to be multiples of `num_cores`.
    tile_counts = [1, 7, num_cores - 1, num_cores + 3, 2 * num_cores + 7, 5 * num_cores - 5]
    shape_list = [tile_size * tiles for tiles in tile_counts]

    torch.manual_seed(0)
    dtype = torch.float32

    for shape in shape_list:
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)

        vec_add_1d_dynamic(x, y, z, x.numel())
        torch.npu.synchronize()

        z_ref = x + y
        torch.testing.assert_close(z, z_ref)
        print(f"result equal for shape {shape}")


if __name__ == "__main__":
    test_add()
