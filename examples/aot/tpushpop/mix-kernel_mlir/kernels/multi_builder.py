from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

PIPE0_ID = 0
PIPE1_ID = 1
SPLIT_UP_DOWN = 1
SLOT_SIZE = 1024
GM_ELEMS_PER_PIPE = 2048  # 8 slots * 1024 bytes / sizeof(f32)
GM_ELEMS_PER_BLOCK = 2 * GM_ELEMS_PER_PIPE
FIFO_BYTES_PER_PIPE = 8 * 1024

ffts_ty = pto.ffts_type
dtype = pto.float32
ptr_ty = pto.PtrType(dtype)
i32 = pto.int32

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


@to_ir_module(module=True)
def module():
    @pto.func(kernel="cube")
    def cube_kernel(gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c16 = const(16)
        c256 = const(256)
        c_pipe_stride = const(GM_ELEMS_PER_PIPE)
        c_block_stride = const(GM_ELEMS_PER_BLOCK)
        c0_i32 = const(0, type=i32)

        # Two independent C2V pipes share one per-block GM backing range.
        # Each pipe owns 8192 bytes, so every AIC/AIV block pair uses 16384
        # bytes. add_ptr offsets are in f32 elements.
        block_idx = s.index_cast(pto.get_block_idx())
        block_num = s.index_cast(pto.get_block_num())
        block_gm_slot_buffer = pto.add_ptr(gm_slot_buffer, block_idx * c_block_stride)
        pipe0_gm_slot_buffer = block_gm_slot_buffer
        pipe1_gm_slot_buffer = pto.add_ptr(block_gm_slot_buffer, c_pipe_stride)
        c2v0_import = pto.import_reserved_buffer(
            name="multi_c2v_fifo_0",
            peer_func="@vector_kernel",
        )
        c2v1_import = pto.import_reserved_buffer(
            name="multi_c2v_fifo_1",
            peer_func="@vector_kernel",
        )

        pto.aic_initialize_pipe(
            id=PIPE0_ID,
            dir_mask=1,
            slot_size=SLOT_SIZE,
            gm_slot_buffer=pipe0_gm_slot_buffer,
            c2v_consumer_buf=c2v0_import,
            v2c_consumer_buf=c0_i32,
        )
        pto.aic_initialize_pipe(
            id=PIPE1_ID,
            dir_mask=1,
            slot_size=SLOT_SIZE,
            gm_slot_buffer=pipe1_gm_slot_buffer,
            c2v_consumer_buf=c2v1_import,
            v2c_consumer_buf=c0_i32,
        )

        x_mat_tile = pto.alloc_tile(x_mat_ty)
        x_left_tile = pto.alloc_tile(x_left_ty)
        x_right_tile = pto.alloc_tile(x_right_ty)
        acc_tile = pto.alloc_tile(acc_ty)

        gm_x_tile_view = pto.slice_view(
            source=pto.as_tensor(
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

        # Send the same tile through two logical pipes. The vector side proves
        # the ids select distinct pipe handles by popping and summing both.
        pto.tpush_to_aiv(acc_tile, SPLIT_UP_DOWN, id=PIPE0_ID)
        pto.tpush_to_aiv(acc_tile, SPLIT_UP_DOWN, id=PIPE1_ID)

    @pto.func(kernel="vector")
    def vector_kernel(gm_slot_buffer: "ptr_ty", gm_y: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c8 = const(8)
        c16 = const(16)
        c256 = const(256)
        c_pipe_stride = const(GM_ELEMS_PER_PIPE)
        c_block_stride = const(GM_ELEMS_PER_BLOCK)
        c0_i32 = const(0, type=i32)

        block_idx = s.index_cast(pto.get_block_idx())
        block_num = s.index_cast(pto.get_block_num())
        block_gm_slot_buffer = pto.add_ptr(gm_slot_buffer, block_idx * c_block_stride)
        pipe0_gm_slot_buffer = block_gm_slot_buffer
        pipe1_gm_slot_buffer = pto.add_ptr(block_gm_slot_buffer, c_pipe_stride)
        c2v0_local = pto.reserve_buffer(
            name="multi_c2v_fifo_0", size=FIFO_BYTES_PER_PIPE, location="VEC"
        )
        c2v1_local = pto.reserve_buffer(
            name="multi_c2v_fifo_1", size=FIFO_BYTES_PER_PIPE, location="VEC"
        )

        pto.aiv_initialize_pipe(
            id=PIPE0_ID,
            dir_mask=1,
            slot_size=SLOT_SIZE,
            gm_slot_buffer=pipe0_gm_slot_buffer,
            c2v_consumer_buf=c2v0_local,
            v2c_consumer_buf=c0_i32,
        )
        pto.aiv_initialize_pipe(
            id=PIPE1_ID,
            dir_mask=1,
            slot_size=SLOT_SIZE,
            gm_slot_buffer=pipe1_gm_slot_buffer,
            c2v_consumer_buf=c2v1_local,
            v2c_consumer_buf=c0_i32,
        )

        subblock_idx = s.index_cast(pto.get_subblock_idx())
        row_offset = subblock_idx * c8

        gm_y_tile_view = pto.slice_view(
            source=pto.as_tensor(
                ptr=gm_y,
                shape=[block_num, c16, c16],
                strides=[c256, c16, c1],
            ),
            offsets=[block_idx, row_offset, c0],
            sizes=[c1, c8, c16],
        )

        recv0_tile = pto.tpop_from_aic(vec_ty, SPLIT_UP_DOWN, id=PIPE0_ID)
        recv1_tile = pto.tpop_from_aic(vec_ty, SPLIT_UP_DOWN, id=PIPE1_ID)
        sum_tile = pto.alloc_tile(vec_ty)
        tile.add(recv0_tile, recv1_tile, sum_tile)
        pto.store(sum_tile, gm_y_tile_view)
        pto.tfree_from_aic(SPLIT_UP_DOWN, id=PIPE0_ID)
        pto.tfree_from_aic(SPLIT_UP_DOWN, id=PIPE1_ID)

    @pto.func
    def call_both(
        ffts_addr: "ffts_ty", gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty", gm_y: "ptr_ty"
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, gm_x)
        pto.call(vector_kernel, gm_slot_buffer, gm_y)


if __name__ == "__main__":
    print(module)
