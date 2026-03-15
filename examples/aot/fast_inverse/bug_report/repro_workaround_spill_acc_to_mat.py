from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const


def meta_data():
    dtype = pto.float32
    i32 = pto.int32
    ptr = pto.PtrType(dtype)
    tv = pto.TensorType(rank=2, dtype=dtype)
    st = pto.SubTensorType(shape=[128, 128], dtype=dtype)
    mat = pto.TileBufType(
        shape=[128, 128], valid_shape=[-1, -1], dtype=dtype, memory_space="MAT"
    )
    left = pto.TileBufType(
        shape=[128, 128], valid_shape=[-1, -1], dtype=dtype, memory_space="LEFT"
    )
    right = pto.TileBufType(
        shape=[128, 128], valid_shape=[-1, -1], dtype=dtype, memory_space="RIGHT"
    )
    acc = pto.TileBufType(
        shape=[128, 128], valid_shape=[-1, -1], dtype=dtype, memory_space="ACC"
    )
    return {
        "ptr": ptr,
        "i32": i32,
        "tv": tv,
        "st": st,
        "mat": mat,
        "left": left,
        "right": right,
        "acc": acc,
    }


@to_ir_module(meta_data=meta_data)
def repro_workaround_spill_acc_to_mat(
    out_ptr: "ptr", in_ptr: "ptr", n_i32: "i32"
) -> None:
    with pto.cube_section():
        c0 = const(0)
        c1 = const(1)
        n = s.index_cast(n_i32)
        x = pto.as_tensor(tv, ptr=in_ptr, shape=[n, n], strides=[n, c1])
        y = pto.as_tensor(tv, ptr=out_ptr, shape=[n, n], strides=[n, c1])
        sx = pto.slice_view(st, source=x, offsets=[c0, c0], sizes=[n, n])
        sy = pto.slice_view(st, source=y, offsets=[c0, c0], sizes=[n, n])
        tmat = pto.alloc_tile(mat, valid_row=n, valid_col=n)
        ta = pto.alloc_tile(left, valid_row=n, valid_col=n)
        tb = pto.alloc_tile(right, valid_row=n, valid_col=n)
        tc = pto.alloc_tile(acc, valid_row=n, valid_col=n)

        pto.load(sx, tmat)
        tile.mov(tmat, ta)
        tile.mov(tmat, tb)
        tile.matmul(ta, tb, tc)

        # Workaround: ACC->GM->MAT instead of ACC->MAT TMOV.
        pto.store(tc, sy)
        pto.load(sy, tmat)

        tile.mov(tmat, ta)
        tile.mov(tmat, tb)
        tile.matmul(ta, tb, tc)
        pto.store(tc, sy)


if __name__ == "__main__":
    print(repro_workaround_spill_acc_to_mat)
