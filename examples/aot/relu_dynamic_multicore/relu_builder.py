from ptodsl import to_ir_module
import ptodsl.language as pto


def build():
    tile_w = 32

    def meta_data():
        dtype = pto.float32
        index_dtype = pto.int32
        ptr_type = pto.PtrType(dtype)
        tensor_type = pto.TensorType(rank=1, dtype=dtype)
        subtensor_type = pto.SubTensorType(shape=[tile_w], dtype=dtype)
        tile_cfg = pto.TileBufConfig()
        # Dynamic valid shape so we can mask partial tiles via valid_row/valid_col.
        tile_type = pto.TileBufType(
            shape=[1, tile_w],
            valid_shape=[-1, -1],
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
            "tile_w": tile_w,
        }

    const = pto.const

    @to_ir_module(meta_data=meta_data)
    def sync_kernel_dyn(arg0: "ptr_type", arg1: "ptr_type", argN: "index_dtype") -> None:
        with pto.vector_section():
            c0 = const(0)
            c1 = const(1)
            c_tile_w = const(tile_w)
            total_elements = pto.index_cast(argN)

            num_blocks = pto.index_cast(pto.get_block_num())
            num_el_per_core = pto.ceil_div(total_elements, num_blocks)

            # Per-core range: [core_start, core_end)
            bid = pto.index_cast(pto.get_block_idx())
            core_start = bid * num_el_per_core
            core_end_unclamped = core_start + num_el_per_core
            core_end = pto.min_u(core_end_unclamped, total_elements)
            core_len = core_end - core_start

            # Per-core number of tiles: ceil(core_len / tile_w).
            num_tiles = pto.ceil_div(core_len, c_tile_w)

            # GM tensors shape N with stride 1.
            tv0 = pto.as_tensor(tensor_type, ptr=arg0, shape=[total_elements], strides=[c1])
            tv1 = pto.as_tensor(tensor_type, ptr=arg1, shape=[total_elements], strides=[c1])

            for i in pto.for_range(c0, num_tiles, c1):
                offset_tile = i * c_tile_w
                offset_total = core_start + offset_tile

                remaining_core = core_end - offset_total
                valid_len = pto.min_u(remaining_core, c_tile_w)

                # Keep per-iteration tile alloc to match original behavior.
                tb0 = pto.alloc_tile(tile_type, valid_row=c1, valid_col=valid_len)
                tb1 = pto.alloc_tile(tile_type, valid_row=c1, valid_col=valid_len)

                # each core c takes a tile at offset c*num_el_per_core + i*tile_w
                sv0 = pto.slice_view(
                    subtensor_type,
                    source=tv0,
                    offsets=[offset_total],
                    sizes=[c_tile_w],
                )
                sv1 = pto.slice_view(
                    subtensor_type,
                    source=tv1,
                    offsets=[offset_total],
                    sizes=[c_tile_w],
                )

                pto.load(sv0, tb0)
                pto.relu(tb0, tb1)
                pto.store(tb1, sv1)

    return sync_kernel_dyn

if __name__ == "__main__":
    print(build())
