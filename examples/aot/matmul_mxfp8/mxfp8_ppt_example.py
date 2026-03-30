from ptodsl import to_ir_module
import ptodsl.language as pto


M, K, N = 16, 64, 32


def meta_data():
    # 1) 选择 MXFP8 组合。默认是 lhs=e5m2, rhs=e5m2, scale=e8m0, acc=f32。
    mx = pto.make_mxfp8(lhs="e5m2", rhs="e5m2")
    scale_k = mx.scale_k(K)  # MXFP8 的 scale 张量沿 K 维按 32:1 压缩

    # 2) 全局输入指针类型
    a_ptr = pto.PtrType(mx.lhs)
    b_ptr = pto.PtrType(mx.rhs)
    scale_ptr = pto.PtrType(mx.scale)

    # 3) TensorView 类型
    a_tensor = pto.TensorType(rank=2, dtype=mx.lhs)
    b_tensor = pto.TensorType(rank=2, dtype=mx.rhs)
    scale_a_tensor = pto.TensorType(rank=2, dtype=mx.scale)
    scale_b_tensor = pto.TensorType(rank=2, dtype=mx.scale)

    # 4) TileView / TileBuf 类型
    a_view = pto.SubTensorType(shape=[M, K], dtype=mx.lhs)
    b_view = pto.SubTensorType(shape=[K, N], dtype=mx.rhs)
    scale_a_view = pto.SubTensorType(shape=[M, scale_k], dtype=mx.scale)
    scale_b_view = pto.SubTensorType(shape=[scale_k, N], dtype=mx.scale)

    a_tile = pto.TileBufType(shape=[M, K], dtype=mx.lhs, memory_space="LEFT")
    b_tile = pto.TileBufType(shape=[K, N], dtype=mx.rhs, memory_space="RIGHT")
    scale_a_tile = pto.LeftScaleTileBufType(shape=[M, scale_k], dtype=mx.scale)
    scale_b_tile = pto.RightScaleTileBufType(shape=[scale_k, N], dtype=mx.scale)
    acc_tile = pto.TileBufType(shape=[M, N], dtype=mx.acc, memory_space="ACC")

    return locals()


@to_ir_module(meta_data=meta_data)
def matmul_mxfp8_core(
    a: "a_ptr",
    scale_a: "scale_ptr",
    b: "b_ptr",
    scale_b: "scale_ptr",
) -> None:
    c0 = pto.const(0)
    c1 = pto.const(1)
    cM = pto.const(M)
    cK = pto.const(K)
    cN = pto.const(N)
    cScaleK = pto.const(scale_k)

    tv_a = pto.as_tensor(a_tensor, ptr=a, shape=[cM, cK], strides=[cK, c1])
    tv_b = pto.as_tensor(b_tensor, ptr=b, shape=[cK, cN], strides=[cN, c1])
    tv_scale_a = pto.as_tensor(scale_a_tensor, ptr=scale_a, shape=[cM, cScaleK], strides=[cScaleK, c1])
    tv_scale_b = pto.as_tensor(scale_b_tensor, ptr=scale_b, shape=[cScaleK, cN], strides=[cN, c1])

    sv_a = pto.slice_view(a_view, source=tv_a, offsets=[c0, c0], sizes=[cM, cK])
    sv_b = pto.slice_view(b_view, source=tv_b, offsets=[c0, c0], sizes=[cK, cN])
    sv_scale_a = pto.slice_view(scale_a_view, source=tv_scale_a, offsets=[c0, c0], sizes=[cM, cScaleK])
    sv_scale_b = pto.slice_view(scale_b_view, source=tv_scale_b, offsets=[c0, c0], sizes=[cScaleK, cN])

    with pto.cube_section():
        ta = pto.alloc_tile(a_tile)
        tb = pto.alloc_tile(b_tile)
        tsa = pto.alloc_tile(scale_a_tile)
        tsb = pto.alloc_tile(scale_b_tile)
        tc = pto.alloc_tile(acc_tile)

        pto.load(sv_a, ta)
        pto.load(sv_b, tb)
        pto.load(sv_scale_a, tsa)
        pto.load(sv_scale_b, tsb)

        # 核心调用：MXFP8 data tile + scale tile -> Acc tile
        pto.matmul_mx(ta, tsa, tb, tsb, tc)


if __name__ == "__main__":
    print(matmul_mxfp8_core)
