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
    y_half_view_ty = pto.SubTensorType(shape=[8, 16], dtype=dtype)
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
    return locals()


@to_ir_module(meta_data=meta_data, module=True)
def module():
    @pto.func(kernel="cube")
    def cube_kernel(gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c16 = const(16)
        c256 = const(256)
        c2048 = const(2048)
        c0_i32 = const(0, type=i32)

        # TPipe does not shard gm_slot_buffer by block id. Give every launched
        # AIC/AIV block pair its own 8-slot C2V FIFO region in GM.
        # slot_size=1024 bytes, slot_num=8 => 8192 bytes per block.
        # addptr offsets are in f32 elements, so 8192 / sizeof(f32) = 2048.
        block_idx = s.index_cast(pto.get_block_idx())
        block_num = s.index_cast(pto.get_block_num())
        block_gm_slot_buffer = pto.addptr(gm_slot_buffer, block_idx * c2048)
        c2v_import = pto.import_reserved_buffer(
            name="c2v_fifo",
            peer_func="@vector_kernel",
        )

        pto.aic_initialize_pipe(
            dir_mask=1,
            slot_size=1024,
            gm_slot_buffer=block_gm_slot_buffer,
            c2v_consumer_buf=c2v_import,
            v2c_consumer_buf=c0_i32,
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

        pto.load(gm_x_tile_view, x_mat_tile)
        tile.mov(x_mat_tile, x_left_tile)
        tile.mov(x_mat_tile, x_right_tile)
        tile.matmul(x_left_tile, x_right_tile, acc_tile)
        # Split the cube result across the two vector subblocks by row.
        pto.tpush_to_aiv(acc_tile, 1)

    @pto.func(kernel="vector")
    def vector_kernel(gm_slot_buffer: "ptr_ty", gm_y: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c8 = const(8)
        c16 = const(16)
        c256 = const(256)
        c2048 = const(2048)
        c0_i32 = const(0, type=i32)

        # Must match cube_kernel's per-block FIFO pointer exactly; otherwise
        # multiple launched blocks would contend for the same GM FIFO slots.
        block_idx = s.index_cast(pto.get_block_idx())
        block_num = s.index_cast(pto.get_block_num())
        block_gm_slot_buffer = pto.addptr(gm_slot_buffer, block_idx * c2048)
        c2v_local = pto.reserve_buffer(name="c2v_fifo", size=8192, location="VEC")

        pto.aiv_initialize_pipe(
            dir_mask=1,
            slot_size=1024,
            gm_slot_buffer=block_gm_slot_buffer,
            c2v_consumer_buf=c2v_local,
            v2c_consumer_buf=c0_i32,
        )

        subblock_idx = s.index_cast(pto.get_subblock_idx())
        row_offset = subblock_idx * c8

        gm_y_tile_view = pto.slice_view(
            y_half_view_ty,
            source=pto.as_tensor(
                tensor_ty,
                ptr=gm_y,
                shape=[block_num, c16, c16],
                strides=[c256, c16, c1],
            ),
            offsets=[block_idx, row_offset, c0],
            sizes=[c1, c8, c16],
        )

        doubled_tile = pto.alloc_tile(vec_ty)
        # split=1 means each vector subblock receives one 8x16 row half of the
        # cube's 16x16 accumulator tile, then stores to its own row slice.
        recv_tile = pto.tpop_from_aic(vec_ty, 1)
        tile.add(recv_tile, recv_tile, doubled_tile)
        pto.store(doubled_tile, gm_y_tile_view)
        pto.tfree_from_aic(1)

    @pto.func
    def call_both(
        ffts_addr: "ffts_ty", gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty", gm_y: "ptr_ty"
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, gm_x)
        pto.call(vector_kernel, gm_slot_buffer, gm_y)


if __name__ == "__main__":
    print(module)
