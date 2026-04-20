from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const


def meta_data():
    ffts_ty = pto.ffts_type
    dtype = pto.float32
    ptr_ty = pto.PtrType(dtype)
    i32 = pto.int32
    tensor_ty = pto.TensorType(rank=3, dtype=dtype)
    tile_view_ty = pto.SubTensorType(shape=[16, 16], dtype=dtype)
    x_half_view_ty = pto.SubTensorType(shape=[8, 16], dtype=dtype)
    vec_ty = pto.TileBufType(shape=[8, 16], dtype=dtype, memory_space="VEC")
    recv_ty = pto.TileBufType(
        shape=[16, 16],
        dtype=dtype,
        memory_space="MAT",
        config=pto.TileBufConfig(
            blayout="RowMajor",
            slayout="NoneBox",
            s_fractal_size=512,
        ),
    )
    return locals()


@to_ir_module(meta_data=meta_data, module=True)
def module():
    @pto.func(kernel="cube")
    def cube_kernel(gm_slot_buffer: "ptr_ty", gm_y: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c16 = const(16)
        c256 = const(256)
        c2048 = const(2048)
        c0_i32 = const(0, type=i32)

        # TPipe does not shard gm_slot_buffer by block id. Give every launched
        # AIC/AIV block pair its own 8-slot V2C FIFO region in GM.
        # slot_size=1024 bytes, slot_num=8 => 8192 bytes per block.
        # addptr offsets are in f32 elements, so 8192 / sizeof(f32) = 2048.
        block_idx = s.index_cast(pto.get_block_idx())
        block_num = s.index_cast(pto.get_block_num())
        block_gm_slot_buffer = pto.addptr(gm_slot_buffer, block_idx * c2048)
        v2c_local = pto.reserve_buffer(name="v2c_fifo", size=8192, location="MAT")

        pto.aic_initialize_pipe(
            dir_mask=2,
            slot_size=1024,
            gm_slot_buffer=block_gm_slot_buffer,
            c2v_consumer_buf=c0_i32,
            v2c_consumer_buf=v2c_local,
        )

        gm_y_tile_view = pto.slice_view(
            tile_view_ty,
            source=pto.as_tensor(
                tensor_ty,
                ptr=gm_y,
                shape=[block_num, c16, c16],
                strides=[c256, c16, c1],
            ),
            offsets=[block_idx, c0, c0],
            sizes=[c1, c16, c16],
        )

        # V2C pop receives the reassembled 16x16 float32 MAT tile from the
        # two vector subblocks' 8x16 float32 halves.
        pto.store(pto.tpop_from_aiv(recv_ty, 1), gm_y_tile_view)
        pto.tfree_from_aiv(1)

    @pto.func(kernel="vector")
    def vector_kernel(gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c8 = const(8)
        c16 = const(16)
        c256 = const(256)
        c2048 = const(2048)
        c0_i32 = const(0, type=i32)

        # Must match cube_kernel's per-block FIFO pointer exactly; otherwise
        # launched block pairs would contend for the same GM FIFO slots.
        block_idx = s.index_cast(pto.get_block_idx())
        block_num = s.index_cast(pto.get_block_num())
        block_gm_slot_buffer = pto.addptr(gm_slot_buffer, block_idx * c2048)
        v2c_import = pto.import_reserved_buffer(
            name="v2c_fifo",
            peer_func="@cube_kernel",
        )

        pto.aiv_initialize_pipe(
            dir_mask=2,
            slot_size=1024,
            gm_slot_buffer=block_gm_slot_buffer,
            c2v_consumer_buf=c0_i32,
            v2c_consumer_buf=v2c_import,
        )

        subblock_idx = s.index_cast(pto.get_subblock_idx())
        row_offset = subblock_idx * c8

        gm_x_tile_view = pto.slice_view(
            x_half_view_ty,
            source=pto.as_tensor(
                tensor_ty,
                ptr=gm_x,
                shape=[block_num, c16, c16],
                strides=[c256, c16, c1],
            ),
            offsets=[block_idx, row_offset, c0],
            sizes=[c1, c8, c16],
        )

        send_tile = pto.alloc_tile(vec_ty)
        pto.load(gm_x_tile_view, send_tile)
        # V2C push sends one 8x16 float32 VEC tile from each vector subblock;
        # split=1 reassembles them into one 16x16 float32 MAT tile on cube.
        pto.tpush_to_aic(send_tile, 1)

    @pto.func
    def call_both(
        ffts_addr: "ffts_ty", gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty", gm_y: "ptr_ty"
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, gm_y)
        pto.call(vector_kernel, gm_slot_buffer, gm_x)


if __name__ == "__main__":
    print(module)
