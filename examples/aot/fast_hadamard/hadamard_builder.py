from ptodsl import to_ir_module
import ptodsl.language as pto

const = pto.const

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


def build_fast_hadamard(fn_name="fast_hadamard_fp16", manual_sync=False):
    """
    Build a dynamic-batch fast-hadamard kernel in PTO DSL.

    Args:
        fn_name: generated kernel symbol name.
        manual_sync:
          - False: rely on `ptoas --enable-insert-sync`.
          - True: emit explicit record/wait events with event_id 0/1.
    """

    @to_ir_module(meta_data=meta_data)
    def _kernel(
        x_ptr: "ptr_type",
        batch_i32: "index_dtype",
        n_i32: "index_dtype",
        log2_n_i32: "index_dtype",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c2 = const(2)
        c_tile = const(ELEMENTS_PER_TILE)

        batch = pto.index_cast(batch_i32)
        n = pto.index_cast(n_i32)
        log2_n = pto.index_cast(log2_n_i32)

        with pto.vector_section():
            # Early reject for invalid n.
            valid_n = pto.gt(n, c0)
            within_tile = pto.ge(c_tile, n)
            with pto.if_context(valid_n):
                with pto.if_context(within_tile):
                    bid = pto.index_cast(pto.get_block_idx())
                    num_blocks = pto.index_cast(pto.get_block_num())

                    # Match reference kernel partitioning: block-level split only.
                    vid = bid
                    num_cores = num_blocks

                    samples_per_core = pto.ceil_div(batch, num_cores)
                    sample_offset = vid * samples_per_core

                    total_elements = batch * n
                    tv_x = pto.as_tensor(
                        tensor_type, ptr=x_ptr, shape=[total_elements], strides=[c1]
                    )

                    # Two independent tile sets (ping/pong) so event_id 0/1 map to
                    # disjoint UB buffers, matching the manual C++ reference.
                    tb_row_0 = pto.alloc_tile(tile_full, valid_col=n)
                    tb_even_0 = pto.alloc_tile(tile_half, valid_col=n // c2)
                    tb_odd_0 = pto.alloc_tile(tile_half, valid_col=n // c2)
                    tb_first_0 = pto.alloc_tile(tile_half, valid_col=n // c2)
                    tb_second_0 = pto.alloc_tile(tile_half, valid_col=n // c2)

                    tb_row_1 = pto.alloc_tile(tile_full, valid_col=n)
                    tb_even_1 = pto.alloc_tile(tile_half, valid_col=n // c2)
                    tb_odd_1 = pto.alloc_tile(tile_half, valid_col=n // c2)
                    tb_first_1 = pto.alloc_tile(tile_half, valid_col=n // c2)
                    tb_second_1 = pto.alloc_tile(tile_half, valid_col=n // c2)

                    with pto.if_context(sample_offset < batch):
                        samples_end = sample_offset + samples_per_core
                        samples_to_process = pto.min_u(samples_end, batch) - sample_offset
                        n_half = n // c2

                        # Keep samples/chunk identical to the C++ idea.
                        samples_per_load = pto.select(n < c_tile, c_tile // n, c1)
                        num_chunks = pto.ceil_div(samples_to_process, samples_per_load)

                        if manual_sync:
                            pto.record_event("VEC", "LOAD", event_id=0)
                            pto.record_event("VEC", "LOAD", event_id=1)
                            pto.record_event("STORE_VEC", "VEC", event_id=0)
                            pto.record_event("STORE_VEC", "VEC", event_id=1)

                        for chunk_i in pto.for_range(c0, num_chunks, c1):
                            sample_done = chunk_i * samples_per_load
                            chunk_left = samples_to_process - sample_done
                            cur_samples = pto.select(
                                chunk_left < samples_per_load, chunk_left, samples_per_load
                            )

                            with pto.if_context(cur_samples > c0):
                                gm_offset = (sample_offset + sample_done) * n
                                use_ev0 = pto.eq(chunk_i % c2, c0)

                                with pto.if_context(use_ev0, has_else=True) as branch:
                                    for s in pto.for_range(c0, cur_samples, c1):
                                        row_offset = gm_offset + s * n
                                        for _ in pto.for_range(c0, log2_n, c1):
                                            sv_row = pto.slice_view(
                                                subtensor_full,
                                                source=tv_x,
                                                offsets=[row_offset],
                                                sizes=[n],
                                            )
                                            sv_first = pto.slice_view(
                                                subtensor_half,
                                                source=tv_x,
                                                offsets=[row_offset],
                                                sizes=[n_half],
                                            )
                                            sv_second = pto.slice_view(
                                                subtensor_half,
                                                source=tv_x,
                                                offsets=[row_offset + n_half],
                                                sizes=[n_half],
                                            )
                                            if manual_sync:
                                                pto.wait_event("VEC", "LOAD", event_id=0)
                                                pto.wait_event(
                                                    "STORE_VEC", "VEC", event_id=0
                                                )
                                            pto.load(sv_row, tb_row_0)
                                            if manual_sync:
                                                pto.record_wait_pair(
                                                    "LOAD", "VEC", event_id=0
                                                )
                                            pto.gather(tb_row_0, tb_even_0, mask_pattern="P0101")
                                            pto.gather(tb_row_0, tb_odd_0, mask_pattern="P1010")
                                            if manual_sync:
                                                pto.barrier("VEC")
                                            pto.add(tb_even_0, tb_odd_0, tb_first_0)
                                            pto.sub(tb_even_0, tb_odd_0, tb_second_0)
                                            if manual_sync:
                                                pto.barrier("VEC")
                                                pto.record_wait_pair(
                                                    "VEC", "STORE_VEC", event_id=0
                                                )
                                            pto.store(tb_first_0, sv_first)
                                            pto.store(tb_second_0, sv_second)
                                            if manual_sync:
                                                pto.record_event(
                                                    "STORE_VEC", "VEC", event_id=0
                                                )
                                                pto.record_event(
                                                    "VEC", "LOAD", event_id=0
                                                )

                                    with branch.else_context():
                                        for s in pto.for_range(c0, cur_samples, c1):
                                            row_offset = gm_offset + s * n
                                            for _ in pto.for_range(c0, log2_n, c1):
                                                sv_row = pto.slice_view(
                                                    subtensor_full,
                                                    source=tv_x,
                                                    offsets=[row_offset],
                                                    sizes=[n],
                                                )
                                                sv_first = pto.slice_view(
                                                    subtensor_half,
                                                    source=tv_x,
                                                    offsets=[row_offset],
                                                    sizes=[n_half],
                                                )
                                                sv_second = pto.slice_view(
                                                    subtensor_half,
                                                    source=tv_x,
                                                    offsets=[row_offset + n_half],
                                                    sizes=[n_half],
                                                )
                                                if manual_sync:
                                                    pto.wait_event(
                                                        "VEC", "LOAD", event_id=1
                                                    )
                                                    pto.wait_event(
                                                        "STORE_VEC", "VEC", event_id=1
                                                    )
                                                pto.load(sv_row, tb_row_1)
                                                if manual_sync:
                                                    pto.record_wait_pair(
                                                        "LOAD", "VEC", event_id=1
                                                    )
                                                pto.gather(
                                                    tb_row_1, tb_even_1, mask_pattern="P0101"
                                                )
                                                pto.gather(
                                                    tb_row_1, tb_odd_1, mask_pattern="P1010"
                                                )
                                                if manual_sync:
                                                    pto.barrier("VEC")
                                                pto.add(tb_even_1, tb_odd_1, tb_first_1)
                                                pto.sub(tb_even_1, tb_odd_1, tb_second_1)
                                                if manual_sync:
                                                    pto.barrier("VEC")
                                                    pto.record_wait_pair(
                                                        "VEC", "STORE_VEC", event_id=1
                                                    )
                                                pto.store(tb_first_1, sv_first)
                                                pto.store(tb_second_1, sv_second)
                                                if manual_sync:
                                                    pto.record_event(
                                                        "STORE_VEC", "VEC", event_id=1
                                                    )
                                                    pto.record_event(
                                                        "VEC", "LOAD", event_id=1
                                                    )

                        if manual_sync:
                            pto.wait_event("VEC", "LOAD", event_id=0)
                            pto.wait_event("VEC", "LOAD", event_id=1)
                            pto.wait_event("STORE_VEC", "VEC", event_id=0)
                            pto.wait_event("STORE_VEC", "VEC", event_id=1)

    # Function name is controlled by the Python function symbol used with
    # to_ir_module; keep fn_name arg for compatibility with caller scripts.
    _ = fn_name
    return _kernel


if __name__ == "__main__":
    # Default: autosync variant, compile with:
    #   ptoas --enable-insert-sync hadamard.pto -o hadamard.cpp
    #
    # Manual sync variant:
    #   python hadamard_builder.py --manual-sync > hadamard.pto
    #   ptoas hadamard.pto -o hadamard.cpp
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Emit explicit record/wait events instead of relying on --enable-insert-sync.",
    )
    parser.add_argument(
        "--fn-name",
        default="fast_hadamard_fp16",
        help="Generated kernel function name.",
    )
    args = parser.parse_args()
    print(build_fast_hadamard(fn_name=args.fn_name, manual_sync=args.manual_sync))
