from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const


def meta_data():
    in_dtype = pto.float16
    out_dtype = pto.float32
    i32 = pto.int32
    ptr_in = pto.PtrType(in_dtype)
    ptr_out = pto.PtrType(out_dtype)
    tv_in = pto.TensorType(rank=2, dtype=in_dtype)
    tv_out = pto.TensorType(rank=2, dtype=out_dtype)
    st_in = pto.SubTensorType(shape=[128, 128], dtype=in_dtype)
    st_out = pto.SubTensorType(shape=[128, 128], dtype=out_dtype)
    mat = pto.TileBufType(
        shape=[128, 128], valid_shape=[-1, -1], dtype=in_dtype, memory_space="MAT"
    )
    left = pto.TileBufType(
        shape=[128, 128], valid_shape=[-1, -1], dtype=in_dtype, memory_space="LEFT"
    )
    right = pto.TileBufType(
        shape=[128, 128], valid_shape=[-1, -1], dtype=in_dtype, memory_space="RIGHT"
    )
    acc = pto.TileBufType(
        shape=[128, 128], valid_shape=[-1, -1], dtype=out_dtype, memory_space="ACC"
    )
    return {
        "ptr_in": ptr_in,
        "ptr_out": ptr_out,
        "i32": i32,
        "tv_in": tv_in,
        "tv_out": tv_out,
        "st_in": st_in,
        "st_out": st_out,
        "mat": mat,
        "left": left,
        "right": right,
        "acc": acc,
    }


@to_ir_module(meta_data=meta_data)
def repro_fail_tmov_acc_to_mat(
    out_ptr: "ptr_out", in_ptr: "ptr_in", n_i32: "i32"
) -> None:
    with pto.cube_section():
        c0 = const(0)
        c1 = const(1)
        n = s.index_cast(n_i32)
        x = pto.as_tensor(tv_in, ptr=in_ptr, shape=[n, n], strides=[n, c1])
        y = pto.as_tensor(tv_out, ptr=out_ptr, shape=[n, n], strides=[n, c1])
        sx = pto.slice_view(st_in, source=x, offsets=[c0, c0], sizes=[n, n])
        sy = pto.slice_view(st_out, source=y, offsets=[c0, c0], sizes=[n, n])
        tmat = pto.alloc_tile(mat, valid_row=n, valid_col=n)
        ta = pto.alloc_tile(left, valid_row=n, valid_col=n)
        tb = pto.alloc_tile(right, valid_row=n, valid_col=n)
        tc = pto.alloc_tile(acc, valid_row=n, valid_col=n)
        pto.load(sx, tmat)
        tile.mov(tmat, ta)
        tile.mov(tmat, tb)
        tile.matmul(ta, tb, tc)
        tile.mov(tc, tmat)  # ACC -> MAT (expected failure)
        pto.store(tc, sy)


if __name__ == "__main__":
    print(repro_fail_tmov_acc_to_mat)
