import os

from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

M = 16
N = 16
TILE_BYTES = M * N * 4
FIFO_SLOTS = 8
FIFO_BYTES = TILE_BYTES * FIFO_SLOTS
NUM_TILES = int(os.environ.get("TPUSHPOP_BENCH_TILES", "8"))

if NUM_TILES <= 0:
    raise ValueError(f"TPUSHPOP_BENCH_TILES must be positive, got {NUM_TILES}.")


def meta_data():
    ffts_ty = pto.ffts_type
    dtype = pto.float32
    ptr_ty = pto.PtrType(dtype)
    i32 = pto.int32
    tensor_ty = pto.TensorType(rank=3, dtype=dtype)
    tile_view_ty = pto.SubTensorType(shape=[M, N], dtype=dtype)
    y_half_view_ty = pto.SubTensorType(shape=[M // 2, N], dtype=dtype)
    x_mat_ty = pto.TileBufType(shape=[M, N], dtype=dtype, memory_space="MAT")
    x_left_ty = pto.TileBufType(
        shape=[M, N],
        dtype=dtype,
        memory_space="LEFT",
        config=pto.TileBufConfig(blayout="ColMajor", slayout="RowMajor"),
    )
    x_right_ty = pto.TileBufType(shape=[M, N], dtype=dtype, memory_space="RIGHT")
    acc_ty = pto.TileBufType(shape=[M, N], dtype=dtype, memory_space="ACC")
    vec_ty = pto.TileBufType(shape=[M // 2, N], dtype=dtype, memory_space="VEC")
    return locals()


@to_ir_module(meta_data=meta_data, module=True)
def module():
    @pto.func(kernel="cube")
    def cube_kernel(gm_slot_buffer: "ptr_ty", gm_x: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c16 = const(16)
        c256 = const(256)
        c_num_tiles = const(NUM_TILES)
        c0_i32 = const(0, type=i32)
        c2v_import = pto.import_reserved_buffer(
            name="c2v_fifo",
            peer_func="@vector_kernel",
        )

        pto.aic_initialize_pipe(
            dir_mask=1,
            slot_size=TILE_BYTES,
            gm_slot_buffer=gm_slot_buffer,
            c2v_consumer_buf=c2v_import,
            v2c_consumer_buf=c0_i32,
        )

        x_mat_tile = pto.alloc_tile(x_mat_ty)
        x_left_tile = pto.alloc_tile(x_left_ty)
        x_right_tile = pto.alloc_tile(x_right_ty)
        acc_tile = pto.alloc_tile(acc_ty)

        gm_x_tensor = pto.as_tensor(
            tensor_ty,
            ptr=gm_x,
            shape=[c_num_tiles, c16, c16],
            strides=[c256, c16, c1],
        )

        for tile_idx in pto.range(c0, c_num_tiles, c1):
            gm_x_tile_view = pto.slice_view(
                tile_view_ty,
                source=gm_x_tensor,
                offsets=[tile_idx, c0, c0],
                sizes=[c1, c16, c16],
            )

            pto.load(gm_x_tile_view, x_mat_tile)
            tile.mov(x_mat_tile, x_left_tile)
            tile.mov(x_mat_tile, x_right_tile)
            tile.matmul(x_left_tile, x_right_tile, acc_tile)
            pto.tpush_to_aiv(acc_tile, 1)

    @pto.func(kernel="vector")
    def vector_kernel(gm_slot_buffer: "ptr_ty", gm_y: "ptr_ty") -> None:
        c0 = const(0)
        c1 = const(1)
        c8 = const(8)
        c16 = const(16)
        c128 = const(128)
        c256 = const(256)
        c_num_tiles = const(NUM_TILES)
        c0_i32 = const(0, type=i32)
        c2v_local = pto.reserve_buffer(
            name="c2v_fifo",
            size=FIFO_BYTES,
            location="VEC",
        )

        pto.aiv_initialize_pipe(
            dir_mask=1,
            slot_size=TILE_BYTES,
            gm_slot_buffer=gm_slot_buffer,
            c2v_consumer_buf=c2v_local,
            v2c_consumer_buf=c0_i32,
        )

        subblock_idx = s.index_cast(pto.get_subblock_idx())
        row_offset = subblock_idx * c8
        gm_y_tensor = pto.as_tensor(
            tensor_ty,
            ptr=gm_y,
            shape=[c_num_tiles, c16, c16],
            strides=[c256, c16, c1],
        )

        doubled_tile = pto.alloc_tile(vec_ty)
        for tile_idx in pto.range(c0, c_num_tiles, c1):
            gm_y_tile_view = pto.slice_view(
                y_half_view_ty,
                source=gm_y_tensor,
                offsets=[tile_idx, row_offset, c0],
                sizes=[c1, c8, c16],
            )

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
