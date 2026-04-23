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
    x_mat_ty = pto.TileBufType(shape=[16, 16], dtype=dtype, memory_space="MAT")
    x_left_ty = pto.TileBufType(
        shape=[16, 16],
        dtype=dtype,
        memory_space="LEFT",
        config=pto.TileBufConfig(blayout="ColMajor", slayout="RowMajor"),
    )
    x_right_ty = pto.TileBufType(shape=[16, 16], dtype=dtype, memory_space="RIGHT")
    acc_ty = pto.TileBufType(shape=[16, 16], dtype=dtype, memory_space="ACC")
    vec_ty = pto.TileBufType(shape=[8, 16], dtype=dtype, memory_space="VEC")
    # Direct GM writeback from cube needs a row-major NoneBox tile.
    cube_recv_ty = pto.TileBufType(
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
    def cube_kernel(gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty", gm_y: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c16 = const(16)
        c256 = const(256)
        c2048 = const(2048)

        # DIR_BOTH uses two 4-slot logical FIFOs. Give every launched
        # AIC/AIV block pair its own 8192-byte GM slot region:
        # 2 directions * 4 slots * 1024 bytes. add_ptr offsets are in f32
        # elements, so 8192 / sizeof(f32) = 2048.
        block_idx = s.index_cast(pto.get_block_idx())
        block_num = s.index_cast(pto.get_block_num())
        block_gm_slot_buffer = pto.add_ptr(gm_slot_buffer, block_idx * c2048)
        c2v_import = pto.import_reserved_buffer(
            name="c2v_fifo",
            peer_func="@vector_kernel",
        )
        v2c_local = pto.reserve_buffer(name="v2c_fifo", size=4096, location="MAT")

        # One DIR_BOTH pipe handles both legs of the round trip.
        pto.aic_initialize_pipe(
            dir_mask=3,
            slot_size=1024,
            gm_slot_buffer=block_gm_slot_buffer,
            c2v_consumer_buf=c2v_import,
            v2c_consumer_buf=v2c_local,
        )

        x_mat_tile = pto.alloc_tile(x_mat_ty)
        x_left_tile = pto.alloc_tile(x_left_ty)
        x_right_tile = pto.alloc_tile(x_right_ty)
        acc_tile = pto.alloc_tile(acc_ty)

        gm_x_tile_view = pto.slice_view(
            tile_view_ty,
            source=pto.as_tensor(
                tensor_ty,
                ptr=gm_x,
                shape=[block_num, c16, c16],
                strides=[c256, c16, c1],
            ),
            offsets=[block_idx, c0, c0],
            sizes=[c1, c16, c16],
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

        pto.load(gm_x_tile_view, x_mat_tile)
        tile.mov(x_mat_tile, x_left_tile)
        tile.mov(x_mat_tile, x_right_tile)
        tile.matmul(x_left_tile, x_right_tile, acc_tile)
        # C2V push sends a 16x16 float32 ACC tile; split=1 delivers one
        # distinct 8x16 float32 half to each vector subblock.
        pto.tpush_to_aiv(acc_tile, 1)
        # V2C pop receives the reassembled 16x16 float32 MAT tile from the
        # two vector subblocks' 8x16 float32 halves.
        returned_tile = pto.tpop_from_aiv(cube_recv_ty, 1)
        pto.store(returned_tile, gm_y_tile_view)
        pto.tfree_from_aiv(1)

    @pto.func(kernel="vector")
    def vector_kernel(gm_slot_buffer: "ptr_ty") -> None:
        c2048 = const(2048)

        # Must match cube_kernel's per-block FIFO pointer exactly; otherwise
        # launched block pairs would contend for the same GM FIFO slots.
        block_idx = s.index_cast(pto.get_block_idx())
        block_gm_slot_buffer = pto.add_ptr(gm_slot_buffer, block_idx * c2048)
        c2v_local = pto.reserve_buffer(name="c2v_fifo", size=4096, location="VEC")
        v2c_import = pto.import_reserved_buffer(
            name="v2c_fifo",
            peer_func="@cube_kernel",
        )

        # Vector pops cube's tile, doubles it, then pushes the result back.
        pto.aiv_initialize_pipe(
            dir_mask=3,
            slot_size=1024,
            gm_slot_buffer=block_gm_slot_buffer,
            c2v_consumer_buf=c2v_local,
            v2c_consumer_buf=v2c_import,
        )

        doubled_tile = pto.alloc_tile(vec_ty)
        # C2V pop receives one 8x16 float32 VEC tile on each vector subblock.
        recv_tile = pto.tpop_from_aic(vec_ty, 1)
        tile.add(recv_tile, recv_tile, doubled_tile)
        # V2C push sends one 8x16 float32 VEC tile from each vector subblock;
        # split=1 reassembles them into one 16x16 float32 MAT tile on cube.
        pto.tpush_to_aic(doubled_tile, 1)
        pto.tfree_from_aic(1)

    @pto.func
    def call_both(
        ffts_addr: "ffts_ty", gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty", gm_y: "ptr_ty"
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, gm_x, gm_y)
        pto.call(vector_kernel, gm_slot_buffer)


if __name__ == "__main__":
    print(module)
