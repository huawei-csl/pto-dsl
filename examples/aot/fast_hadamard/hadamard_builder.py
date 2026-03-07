from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

ELEMENTS_PER_TILE = 32 * 1024 // 2  # 32KB UB / sizeof(fp16)
HALF_ELEMENTS_PER_TILE = ELEMENTS_PER_TILE // 2


def meta_data():
    dtype = pto.float16
    ptr_type = pto.PtrType(dtype)
    index_dtype = pto.int32

    tensor_type = pto.TensorType(rank=1, dtype=dtype)
    subtensor_full = pto.SubTensorType(shape=[1, ELEMENTS_PER_TILE], dtype=dtype)
    subtensor_half = pto.SubTensorType(shape=[1, HALF_ELEMENTS_PER_TILE], dtype=dtype)

    tile_cfg = pto.TileBufConfig()
    tile_full = pto.TileBufType(
        shape=[1, ELEMENTS_PER_TILE],
        valid_shape=[1, -1],
        dtype=dtype,
        memory_space="VEC",
        config=tile_cfg,
    )
    tile_half = pto.TileBufType(
        shape=[1, HALF_ELEMENTS_PER_TILE],
        valid_shape=[1, -1],
        dtype=dtype,
        memory_space="VEC",
        config=tile_cfg,
    )

    return {
        "ptr_type": ptr_type,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_full": subtensor_full,
        "subtensor_half": subtensor_half,
        "tile_full": tile_full,
        "tile_half": tile_half,
    }


@to_ir_module(meta_data=meta_data)
def fast_hadamard_autosync(
    x_ptr: "ptr_type",
    batch_i32: "index_dtype",
    n_i32: "index_dtype",
    log2_n_i32: "index_dtype",
) -> None:
    c0 = const(0)
    c1 = const(1)
    c2 = const(2)

    batch = s.index_cast(batch_i32)
    n = s.index_cast(n_i32)
    log2_n = s.index_cast(log2_n_i32)

    cid = pto.get_block_idx()
    sub_bid = pto.get_subblock_idx()
    sub_bnum = pto.get_subblock_num()
    num_blocks = pto.get_block_num()

    vid = s.index_cast(cid * sub_bnum + sub_bid)  # vector core index
    num_cores = s.index_cast(num_blocks * sub_bnum)  # number of vector cores

    with pto.vector_section():
        samples_per_core = s.ceil_div(batch, num_cores)
        sample_offset = vid * samples_per_core

        with pto.if_context(sample_offset < batch):
            samples_end = sample_offset + samples_per_core
            samples_to_process = s.select(
                samples_end > batch,
                batch - sample_offset,
                samples_per_core,
            )

            with pto.if_context(samples_to_process > c0):
                total_elements = batch * n
                tv_x = pto.as_tensor(
                    tensor_type, ptr=x_ptr, shape=[total_elements], strides=[c1]
                )

                # Two independent tile sets (ping/pong) so event_id 0/1 map to
                # disjoint UB buffers, matching the manual C++ reference.
                tb_row_0 = pto.alloc_tile(tile_full, valid_col=n)
                tb_even_0 = pto.alloc_tile(tile_half, valid_col=n // c2)
                tb_odd_0 = pto.alloc_tile(tile_half, valid_col=n // c2)

                tb_row_1 = pto.alloc_tile(tile_full, valid_col=n)
                tb_even_1 = pto.alloc_tile(tile_half, valid_col=n // c2)
                tb_odd_1 = pto.alloc_tile(tile_half, valid_col=n // c2)

                n_half = n // c2

                # Keep one sample per chunk. Multi-sample chunks interact
                # poorly with static tile subset sizing in current PTO Python
                # bindings and can corrupt rows for larger batches.
                samples_per_load = c1
                num_chunks = s.ceil_div(samples_to_process, samples_per_load)

                def process_rows(tb_row, tb_even, tb_odd, gm_offset, cur_samples):
                    for s in pto.for_range(c0, cur_samples, c1):
                        row_offset = gm_offset + s * n
                        sv_row = pto.slice_view(
                            subtensor_full, source=tv_x, offsets=[row_offset], sizes=[n]
                        )
                        # Alias row halves inside UB row tile (no GM round-trip
                        # per Hadamard iteration).
                        tb_first = tile.subset(tb_row, [c0, c0], [1, HALF_ELEMENTS_PER_TILE])
                        tb_second = tile.subset(tb_row, [c0, n_half], [1, HALF_ELEMENTS_PER_TILE])

                        pto.load(sv_row, tb_row)
                        for _ in pto.for_range(c0, log2_n, c1):
                            tile.gather(tb_row, tb_even, mask_pattern="P0101")
                            tile.gather(tb_row, tb_odd, mask_pattern="P1010")
                            tile.add(tb_even, tb_odd, tb_first)
                            tile.sub(tb_even, tb_odd, tb_second)
                        pto.store(tb_row, sv_row)

                for chunk_i in pto.for_range(c0, num_chunks, c1):
                    sample_done = chunk_i * samples_per_load
                    chunk_left = samples_to_process - sample_done
                    cur_samples = s.select(
                        chunk_left < samples_per_load, chunk_left, samples_per_load
                    )

                    with pto.if_context(cur_samples > c0):
                        gm_offset = (sample_offset + sample_done) * n
                        use_ev0 = (chunk_i % c2) == c0

                        with pto.if_context(use_ev0, has_else=True) as branch:
                            process_rows(
                                tb_row_0, tb_even_0, tb_odd_0, gm_offset, cur_samples
                            )
                        with branch.else_context():
                            process_rows(
                                tb_row_1, tb_even_1, tb_odd_1, gm_offset, cur_samples
                            )


@to_ir_module(meta_data=meta_data)
def fast_hadamard_manualsync(
    x_ptr: "ptr_type",
    batch_i32: "index_dtype",
    n_i32: "index_dtype",
    log2_n_i32: "index_dtype",
) -> None:
    c0 = const(0)
    c1 = const(1)
    c2 = const(2)

    batch = s.index_cast(batch_i32)
    n = s.index_cast(n_i32)
    log2_n = s.index_cast(log2_n_i32)

    cid = pto.get_block_idx()
    sub_bid = pto.get_subblock_idx()
    sub_bnum = pto.get_subblock_num()
    num_blocks = pto.get_block_num()

    vid = s.index_cast(cid * sub_bnum + sub_bid)  # vector core index
    num_cores = s.index_cast(num_blocks * sub_bnum)  # number of vector cores

    with pto.vector_section():
        samples_per_core = s.ceil_div(batch, num_cores)
        sample_offset = vid * samples_per_core

        with pto.if_context(sample_offset < batch):
            samples_end = sample_offset + samples_per_core
            samples_to_process = s.select(
                samples_end > batch,
                batch - sample_offset,
                samples_per_core,
            )

            with pto.if_context(samples_to_process > c0):
                total_elements = batch * n
                tv_x = pto.as_tensor(
                    tensor_type, ptr=x_ptr, shape=[total_elements], strides=[c1]
                )

                # Two independent tile sets (ping/pong) so event_id 0/1 map to
                # disjoint UB buffers, matching the manual C++ reference.
                tb_row_0 = pto.alloc_tile(tile_full, valid_col=n)
                tb_even_0 = pto.alloc_tile(tile_half, valid_col=n // c2)
                tb_odd_0 = pto.alloc_tile(tile_half, valid_col=n // c2)

                tb_row_1 = pto.alloc_tile(tile_full, valid_col=n)
                tb_even_1 = pto.alloc_tile(tile_half, valid_col=n // c2)
                tb_odd_1 = pto.alloc_tile(tile_half, valid_col=n // c2)

                n_half = n // c2

                # Keep one sample per chunk. Multi-sample chunks interact
                # poorly with static tile subset sizing in current PTO Python
                # bindings and can corrupt rows for larger batches.
                samples_per_load = c1
                num_chunks = s.ceil_div(samples_to_process, samples_per_load)

                def process_rows(
                    tb_row, tb_even, tb_odd, event_id, gm_offset, cur_samples
                ):
                    for s in pto.for_range(c0, cur_samples, c1):
                        row_offset = gm_offset + s * n
                        sv_row = pto.slice_view(
                            subtensor_full, source=tv_x, offsets=[row_offset], sizes=[n]
                        )
                        # Alias row halves inside UB row tile (no GM round-trip
                        # per Hadamard iteration).
                        tb_first = tile.subset(
                            tb_row, [c0, c0], [1, HALF_ELEMENTS_PER_TILE]
                        )
                        tb_second = tile.subset(
                            tb_row, [c0, n_half], [1, HALF_ELEMENTS_PER_TILE]
                        )

                        pto.wait_event("VEC", "LOAD", event_id=event_id)
                        pto.wait_event("STORE_VEC", "VEC", event_id=event_id)
                        pto.load(sv_row, tb_row)
                        pto.record_wait_pair("LOAD", "VEC", event_id=event_id)

                        for _ in pto.for_range(c0, log2_n, c1):
                            tile.gather(tb_row, tb_even, mask_pattern="P0101")
                            tile.gather(tb_row, tb_odd, mask_pattern="P1010")
                            pto.barrier("VEC")
                            tile.add(tb_even, tb_odd, tb_first)
                            tile.sub(tb_even, tb_odd, tb_second)
                            pto.barrier("VEC")

                        pto.record_wait_pair(
                            "VEC", "STORE_VEC", event_id=event_id
                        )
                        pto.store(tb_row, sv_row)
                        pto.record_event("STORE_VEC", "VEC", event_id=event_id)
                        pto.record_event("VEC", "LOAD", event_id=event_id)

                for event_id in (0, 1):
                    pto.record_event("VEC", "LOAD", event_id=event_id)
                    pto.record_event("STORE_VEC", "VEC", event_id=event_id)

                for chunk_i in pto.for_range(c0, num_chunks, c1):
                    sample_done = chunk_i * samples_per_load
                    chunk_left = samples_to_process - sample_done
                    cur_samples = s.select(
                        chunk_left < samples_per_load, chunk_left, samples_per_load
                    )

                    with pto.if_context(cur_samples > c0):
                        gm_offset = (sample_offset + sample_done) * n
                        use_ev0 = (chunk_i % c2) == c0

                        with pto.if_context(use_ev0, has_else=True) as branch:
                            process_rows(tb_row_0, tb_even_0, tb_odd_0, 0, gm_offset, cur_samples)
                        with branch.else_context():
                            process_rows(tb_row_1, tb_even_1, tb_odd_1, 1, gm_offset, cur_samples)

                for event_id in (0, 1):
                    pto.wait_event("VEC", "LOAD", event_id=event_id)
                    pto.wait_event("STORE_VEC", "VEC", event_id=event_id)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Emit explicit record/wait events instead of relying on --enable-insert-sync.",
    )
    args = parser.parse_args()
    if args.manual_sync:
        module = fast_hadamard_manualsync
    else:
        module = fast_hadamard_autosync
    print(module)
