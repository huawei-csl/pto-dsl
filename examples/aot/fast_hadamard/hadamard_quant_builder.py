from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

ELEMENTS_PER_TILE = 32 * 1024 // 2  # 32 KB UB / sizeof(fp16)
HALF_ELEMENTS_PER_TILE = ELEMENTS_PER_TILE // 2


def meta_data():
    f16 = pto.float16
    i8 = pto.int8
    tile_cfg = pto.TileBufConfig()
    return {
        "in_ptr": pto.PtrType(f16),
        "out_ptr": pto.PtrType(i8),
        "i32_t": pto.int32,
        "f32_t": pto.float32,
        "in_tensor": pto.TensorType(rank=1, dtype=f16),
        "out_tensor": pto.TensorType(rank=1, dtype=i8),
        "sub_full_in": pto.SubTensorType(shape=[1, ELEMENTS_PER_TILE], dtype=f16),
        "sub_full_out": pto.SubTensorType(shape=[1, ELEMENTS_PER_TILE], dtype=i8),
        "tile_full_in": pto.TileBufType(
            shape=[1, ELEMENTS_PER_TILE],
            valid_shape=[1, -1],
            dtype=f16,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_half_in": pto.TileBufType(
            shape=[1, HALF_ELEMENTS_PER_TILE],
            valid_shape=[1, -1],
            dtype=f16,
            memory_space="VEC",
            config=tile_cfg,
        ),
        "tile_full_out": pto.TileBufType(
            shape=[1, ELEMENTS_PER_TILE],
            valid_shape=[1, -1],
            dtype=i8,
            memory_space="VEC",
            config=tile_cfg,
        ),
    }


def build_fast_hadamard_quant_autosync(group_size=None):
    """Build a fused Hadamard+quantize kernel (fp16 input → int8 output).

    Sync is handled automatically by --enable-insert-sync.

    Args:
        group_size: Elements per quantization group.  Must evenly divide n.
                    Pass None for a uniform-quantization kernel (no per-group loops).

    Kernel parameters (runtime):
        x_ptr              – fp16 input  [batch, n]
        y_ptr              – int8 output [batch, n]
        group_scales_ptr   – fp16 per-group scales  (ignored when group_size is None)
        group_offsets_ptr  – fp16 per-group offsets (ignored when group_size is None)
        scale_group_stride_i32  – row stride into group_scales (0 = shared across rows)
        offset_group_stride_i32 – row stride into group_offsets (0 = shared across rows)
        batch_i32, n_i32, log2_n_i32 – tensor dimensions
        scale_f32    – uniform scale  (fallback when has_group_scales == 0)
        group_size_i32 – (informational, not used for tile sizing when group_size is set)
        q_offset_f32 – uniform quant offset (fallback when has_group_offsets == 0)
        has_group_scales_i32  – 1 if group_scales_ptr holds valid data, else 0
        has_group_offsets_i32 – 1 if group_offsets_ptr holds valid data, else 0
    """
    use_groups = group_size is not None

    @to_ir_module(meta_data=meta_data)
    def fast_hadamard_quant_autosync(
        x_ptr: "in_ptr",
        y_ptr: "out_ptr",
        group_scales_ptr: "in_ptr",
        group_offsets_ptr: "in_ptr",
        scale_group_stride_i32: "i32_t",
        offset_group_stride_i32: "i32_t",
        batch_i32: "i32_t",
        n_i32: "i32_t",
        log2_n_i32: "i32_t",
        scale_f32: "f32_t",
        group_size_i32: "i32_t",
        q_offset_f32: "f32_t",
        has_group_scales_i32: "i32_t",
        has_group_offsets_i32: "i32_t",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c2 = const(2)

        batch = s.index_cast(batch_i32)
        n = s.index_cast(n_i32)
        log2_n = s.index_cast(log2_n_i32)
        scale_group_stride = s.index_cast(scale_group_stride_i32)
        offset_group_stride = s.index_cast(offset_group_stride_i32)

        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        num_blocks = pto.get_block_num()

        vid = s.index_cast(cid * sub_bnum + sub_bid)
        num_cores = s.index_cast(num_blocks * sub_bnum)

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
                    tv_x = pto.as_tensor(
                        in_tensor, ptr=x_ptr, shape=[batch * n], strides=[c1]
                    )
                    tv_y = pto.as_tensor(
                        out_tensor, ptr=y_ptr, shape=[batch * n], strides=[c1]
                    )

                    tb_x_0 = pto.alloc_tile(tile_full_in, valid_col=n)
                    tb_x_1 = pto.alloc_tile(tile_full_in, valid_col=n)
                    tb_y_0 = pto.alloc_tile(tile_full_out, valid_col=n)
                    tb_y_1 = pto.alloc_tile(tile_full_out, valid_col=n)

                    tb_even = pto.alloc_tile(tile_half_in, valid_col=n // c2)
                    tb_odd = pto.alloc_tile(tile_half_in, valid_col=n // c2)

                    n_half = n // c2

                    scale_f16 = s.truncf(scale_f32, s.float16)
                    q_offset_f16 = s.truncf(q_offset_f32, s.float16)

                    has_gs = s.index_cast(has_group_scales_i32) != c0
                    has_go = s.index_cast(has_group_offsets_i32) != c0

                    samples_per_load = c1
                    num_chunks = s.ceil_div(samples_to_process, samples_per_load)

                    def process_chunk(tb_x, tb_y, event_id, gm_offset):
                        sv_x = pto.slice_view(
                            sub_full_in,
                            source=tv_x,
                            offsets=[gm_offset],
                            sizes=[n],
                        )
                        sv_y = pto.slice_view(
                            sub_full_out,
                            source=tv_y,
                            offsets=[gm_offset],
                            sizes=[n],
                        )

                        pto.load(sv_x, tb_x)

                        # ── Hadamard transform (in-place on tb_x) ─────────
                        tb_first = tile.subview(
                            tb_x, [c0, c0], [1, HALF_ELEMENTS_PER_TILE]
                        )
                        tb_second = tile.subview(
                            tb_x, [c0, n_half], [1, HALF_ELEMENTS_PER_TILE]
                        )

                        for _ in pto.range(c0, log2_n, c1):
                            tile.gather(tb_x, tb_even, mask_pattern="P0101")
                            tile.gather(tb_x, tb_odd, mask_pattern="P1010")
                            tile.add(tb_even, tb_odd, tb_first)
                            tile.sub(tb_even, tb_odd, tb_second)

                        # ── quantize ──────────────────────────────────────
                        if use_groups:
                            c_gs = const(group_size)
                            row_index = gm_offset // n
                            groups_per_row = n // c_gs

                            for g in pto.range(c0, groups_per_row, c1):
                                g_off = g * c_gs
                                xg = tile.subview(tb_x, [c0, g_off], [1, group_size])
                                yg = tile.subview(tb_y, [c0, g_off], [1, group_size])

                                with pto.if_context(has_gs, has_else=True) as sbr:
                                    scale_idx = s.select(
                                        scale_group_stride == c0,
                                        g,
                                        row_index * scale_group_stride + g,
                                    )
                                    gs = pto.load_scalar(
                                        s.float16, group_scales_ptr, scale_idx
                                    )
                                    tile.muls(xg, gs, xg)
                                with sbr.else_context():
                                    tile.muls(xg, scale_f16, xg)

                                with pto.if_context(has_go, has_else=True) as obr:
                                    offset_idx = s.select(
                                        offset_group_stride == c0,
                                        g,
                                        row_index * offset_group_stride + g,
                                    )
                                    go = pto.load_scalar(
                                        s.float16, group_offsets_ptr, offset_idx
                                    )
                                    tile.adds(xg, go, xg)
                                with obr.else_context():
                                    tile.adds(xg, q_offset_f16, xg)

                                tile.cvt(xg, yg, rmode="none")

                        else:
                            tile.muls(tb_x, scale_f16, tb_x)
                            tile.adds(tb_x, q_offset_f16, tb_x)
                            tile.cvt(tb_x, tb_y, rmode="none")

                        pto.store(tb_y, sv_y)

                    for chunk_i in pto.range(c0, num_chunks, c1):
                        sample_done = chunk_i * samples_per_load
                        chunk_left = samples_to_process - sample_done
                        cur_samples = s.select(
                            chunk_left < samples_per_load,
                            chunk_left,
                            samples_per_load,
                        )

                        with pto.if_context(cur_samples > c0):
                            gm_offset = (sample_offset + sample_done) * n
                            use_ev0 = (chunk_i % c2) == c0

                            with pto.if_context(use_ev0, has_else=True) as br:
                                process_chunk(tb_x_0, tb_y_0, 0, gm_offset)
                            with br.else_context():
                                process_chunk(tb_x_1, tb_y_1, 1, gm_offset)

    return fast_hadamard_quant_autosync


def build_fast_hadamard_quant_manualsync(group_size=None):
    """Build a fused Hadamard+quantize kernel (fp16 input → int8 output).

    Emits explicit record/wait events instead of relying on --enable-insert-sync.

    Args:
        group_size: Elements per quantization group.  Must evenly divide n.
                    Pass None for a uniform-quantization kernel (no per-group loops).

    Kernel parameters (runtime): same as build_fast_hadamard_quant_autosync.
    """
    use_groups = group_size is not None

    @to_ir_module(meta_data=meta_data)
    def fast_hadamard_quant_manualsync(
        x_ptr: "in_ptr",
        y_ptr: "out_ptr",
        group_scales_ptr: "in_ptr",
        group_offsets_ptr: "in_ptr",
        scale_group_stride_i32: "i32_t",
        offset_group_stride_i32: "i32_t",
        batch_i32: "i32_t",
        n_i32: "i32_t",
        log2_n_i32: "i32_t",
        scale_f32: "f32_t",
        group_size_i32: "i32_t",
        q_offset_f32: "f32_t",
        has_group_scales_i32: "i32_t",
        has_group_offsets_i32: "i32_t",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        c2 = const(2)

        batch = s.index_cast(batch_i32)
        n = s.index_cast(n_i32)
        log2_n = s.index_cast(log2_n_i32)
        scale_group_stride = s.index_cast(scale_group_stride_i32)
        offset_group_stride = s.index_cast(offset_group_stride_i32)

        cid = pto.get_block_idx()
        sub_bid = pto.get_subblock_idx()
        sub_bnum = pto.get_subblock_num()
        num_blocks = pto.get_block_num()

        vid = s.index_cast(cid * sub_bnum + sub_bid)
        num_cores = s.index_cast(num_blocks * sub_bnum)

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
                    tv_x = pto.as_tensor(
                        in_tensor, ptr=x_ptr, shape=[batch * n], strides=[c1]
                    )
                    tv_y = pto.as_tensor(
                        out_tensor, ptr=y_ptr, shape=[batch * n], strides=[c1]
                    )

                    tb_x_0 = pto.alloc_tile(tile_full_in, valid_col=n)
                    tb_x_1 = pto.alloc_tile(tile_full_in, valid_col=n)
                    tb_y_0 = pto.alloc_tile(tile_full_out, valid_col=n)
                    tb_y_1 = pto.alloc_tile(tile_full_out, valid_col=n)

                    tb_even = pto.alloc_tile(tile_half_in, valid_col=n // c2)
                    tb_odd = pto.alloc_tile(tile_half_in, valid_col=n // c2)

                    n_half = n // c2

                    scale_f16 = s.truncf(scale_f32, s.float16)
                    q_offset_f16 = s.truncf(q_offset_f32, s.float16)

                    has_gs = s.index_cast(has_group_scales_i32) != c0
                    has_go = s.index_cast(has_group_offsets_i32) != c0

                    samples_per_load = c1
                    num_chunks = s.ceil_div(samples_to_process, samples_per_load)

                    def process_chunk(tb_x, tb_y, event_id, gm_offset):
                        sv_x = pto.slice_view(
                            sub_full_in,
                            source=tv_x,
                            offsets=[gm_offset],
                            sizes=[n],
                        )
                        sv_y = pto.slice_view(
                            sub_full_out,
                            source=tv_y,
                            offsets=[gm_offset],
                            sizes=[n],
                        )

                        # ── load ──────────────────────────────────────────
                        pto.wait_event("VEC", "LOAD", event_id=event_id)
                        pto.wait_event("STORE_VEC", "VEC", event_id=event_id)
                        pto.load(sv_x, tb_x)
                        pto.record_wait_pair("LOAD", "VEC", event_id=event_id)

                        # ── Hadamard transform (in-place on tb_x) ─────────
                        tb_first = tile.subview(
                            tb_x, [c0, c0], [1, HALF_ELEMENTS_PER_TILE]
                        )
                        tb_second = tile.subview(
                            tb_x, [c0, n_half], [1, HALF_ELEMENTS_PER_TILE]
                        )

                        for _ in pto.range(c0, log2_n, c1):
                            tile.gather(tb_x, tb_even, mask_pattern="P0101")
                            tile.gather(tb_x, tb_odd, mask_pattern="P1010")
                            pto.barrier("VEC")
                            tile.add(tb_even, tb_odd, tb_first)
                            tile.sub(tb_even, tb_odd, tb_second)
                            pto.barrier("VEC")

                        # ── quantize ──────────────────────────────────────
                        if use_groups:
                            c_gs = const(group_size)
                            row_index = gm_offset // n
                            groups_per_row = n // c_gs

                            for g in pto.range(c0, groups_per_row, c1):
                                g_off = g * c_gs
                                xg = tile.subview(tb_x, [c0, g_off], [1, group_size])
                                yg = tile.subview(tb_y, [c0, g_off], [1, group_size])

                                with pto.if_context(has_gs, has_else=True) as sbr:
                                    scale_idx = s.select(
                                        scale_group_stride == c0,
                                        g,
                                        row_index * scale_group_stride + g,
                                    )
                                    gs = pto.load_scalar(
                                        s.float16, group_scales_ptr, scale_idx
                                    )
                                    tile.muls(xg, gs, xg)
                                with sbr.else_context():
                                    tile.muls(xg, scale_f16, xg)

                                pto.barrier("VEC")

                                with pto.if_context(has_go, has_else=True) as obr:
                                    offset_idx = s.select(
                                        offset_group_stride == c0,
                                        g,
                                        row_index * offset_group_stride + g,
                                    )
                                    go = pto.load_scalar(
                                        s.float16, group_offsets_ptr, offset_idx
                                    )
                                    tile.adds(xg, go, xg)
                                with obr.else_context():
                                    tile.adds(xg, q_offset_f16, xg)

                                pto.barrier("VEC")

                                tile.cvt(xg, yg, rmode="none")

                                pto.barrier("VEC")

                        else:
                            tile.muls(tb_x, scale_f16, tb_x)
                            pto.barrier("VEC")
                            tile.adds(tb_x, q_offset_f16, tb_x)
                            pto.barrier("VEC")
                            tile.cvt(tb_x, tb_y, rmode="none")
                            pto.barrier("VEC")

                        # ── store ─────────────────────────────────────────
                        pto.record_wait_pair("VEC", "STORE_VEC", event_id=event_id)
                        pto.store(tb_y, sv_y)
                        pto.record_event("STORE_VEC", "VEC", event_id=event_id)
                        pto.record_event("VEC", "LOAD", event_id=event_id)

                    # Seed the fence state so the first chunk can proceed.
                    for ev in (0, 1):
                        pto.record_event("VEC", "LOAD", event_id=ev)
                        pto.record_event("STORE_VEC", "VEC", event_id=ev)

                    for chunk_i in pto.range(c0, num_chunks, c1):
                        sample_done = chunk_i * samples_per_load
                        chunk_left = samples_to_process - sample_done
                        cur_samples = s.select(
                            chunk_left < samples_per_load,
                            chunk_left,
                            samples_per_load,
                        )

                        with pto.if_context(cur_samples > c0):
                            gm_offset = (sample_offset + sample_done) * n
                            use_ev0 = (chunk_i % c2) == c0

                            with pto.if_context(use_ev0, has_else=True) as br:
                                process_chunk(tb_x_0, tb_y_0, 0, gm_offset)
                            with br.else_context():
                                process_chunk(tb_x_1, tb_y_1, 1, gm_offset)

                    for ev in (0, 1):
                        pto.wait_event("VEC", "LOAD", event_id=ev)
                        pto.wait_event("STORE_VEC", "VEC", event_id=ev)

    return fast_hadamard_quant_manualsync


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Per-group quantization group size.  Omit for uniform quantization.",
    )
    parser.add_argument(
        "--manual-sync",
        action="store_true",
        help="Emit explicit record/wait events instead of relying on --enable-insert-sync.",
    )
    args = parser.parse_args()

    if args.manual_sync:
        module = build_fast_hadamard_quant_manualsync(group_size=args.group_size)
    else:
        module = build_fast_hadamard_quant_autosync(group_size=args.group_size)
    print(module)
