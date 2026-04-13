from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const


def meta_data():
    ffts_ty = pto.ffts_type
    dtype = pto.float32
    ptr_ty = pto.PtrType(dtype)
    i32 = pto.int32
    tensor_ty = pto.TensorType(rank=2, dtype=dtype)
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
    recv_ty = pto.TileBufType(shape=[16, 16], dtype=dtype, memory_space="VEC")
    return locals()


@to_ir_module(meta_data=meta_data, module=True)
def module():
    @pto.func(kernel="cube")
    def cube_kernel(gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c16 = const(16)
        c0_i32 = const(0, type=i32)
        c2v_import = pto.import_reserved_buffer(
            name="c2v_fifo",
            peer_func="@vector_kernel",
        )

        pto.aic_initialize_pipe(
            dir_mask=1,
            slot_size=1024,
            gm_slot_buffer=gm_slot_buffer,
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
                shape=[c16, c16],
                strides=[c16, c1],
            ),
            offsets=[c0, c0],
            sizes=[c16, c16],
        )

        pto.load(gm_x_tile_view, x_mat_tile)
        tile.mov(x_mat_tile, x_left_tile)
        tile.mov(x_mat_tile, x_right_tile)
        tile.matmul(x_left_tile, x_right_tile, acc_tile)
        pto.tpush_to_aiv(acc_tile, 0)

    @pto.func(kernel="vector")
    def vector_kernel(gm_slot_buffer: "ptr_ty", gm_y: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c16 = const(16)
        c0_i32 = const(0, type=i32)
        c2v_local = pto.reserve_buffer(name="c2v_fifo", size=4096, location="VEC")

        pto.aiv_initialize_pipe(
            dir_mask=1,
            slot_size=1024,
            gm_slot_buffer=gm_slot_buffer,
            c2v_consumer_buf=c2v_local,
            v2c_consumer_buf=c0_i32,
        )

        gm_y_tile_view = pto.slice_view(
            tile_view_ty,
            source=pto.as_tensor(
                tensor_ty,
                ptr=gm_y,
                shape=[c16, c16],
                strides=[c16, c1],
            ),
            offsets=[c0, c0],
            sizes=[c16, c16],
        )

        pto.store(pto.tpop_from_aic(recv_ty, 0), gm_y_tile_view)
        pto.tfree_from_aic(0)

    @pto.func
    def call_both(
        ffts_addr: "ffts_ty", gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty", gm_y: "ptr_ty"
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, gm_x)
        pto.call(vector_kernel, gm_slot_buffer, gm_y)


if __name__ == "__main__":
    print(module)
