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
    vec_ty = pto.TileBufType(shape=[16, 16], dtype=dtype, memory_space="VEC")
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
        c0_i32 = const(0, type=i32)
        v2c_local = pto.reserve_buffer(name="v2c_fifo", size=4096, location="MAT")

        pto.aic_initialize_pipe(
            dir_mask=2,
            slot_size=1024,
            gm_slot_buffer=gm_slot_buffer,
            c2v_consumer_buf=c0_i32,
            v2c_consumer_buf=v2c_local,
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

        pto.store(pto.tpop_from_aiv(recv_ty, 0), gm_y_tile_view)
        pto.tfree_from_aiv(0)

    @pto.func(kernel="vector")
    def vector_kernel(gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c16 = const(16)
        c0_i32 = const(0, type=i32)
        v2c_import = pto.import_reserved_buffer(
            name="v2c_fifo",
            peer_func="@cube_kernel",
        )

        pto.aiv_initialize_pipe(
            dir_mask=2,
            slot_size=1024,
            gm_slot_buffer=gm_slot_buffer,
            c2v_consumer_buf=c0_i32,
            v2c_consumer_buf=v2c_import,
        )

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

        send_tile = pto.alloc_tile(vec_ty)
        pto.load(gm_x_tile_view, send_tile)
        pto.tpush_to_aic(send_tile, 0)

    @pto.func
    def call_both(
        ffts_addr: "ffts_ty", gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty", gm_y: "ptr_ty"
    ) -> None:
        pto.set_ffts(ffts_addr)
        pto.call(cube_kernel, gm_slot_buffer, gm_y)
        pto.call(vector_kernel, gm_slot_buffer, gm_x)


if __name__ == "__main__":
    print(module)
