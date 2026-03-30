from ptodsl import to_ir_module
import ptodsl.language as pto


def build(M=16, K=64, N=32, lhs_variant="e5m2", rhs_variant="e5m2"):
    def meta_data():
        mx = pto.make_mxfp8(lhs=lhs_variant, rhs=rhs_variant)
        scale_k = mx.scale_k(K)

        ptr_lhs = pto.PtrType(mx.lhs)
        ptr_rhs = pto.PtrType(mx.rhs)
        ptr_scale = pto.PtrType(mx.scale)
        ptr_bias = pto.PtrType(mx.acc)

        lhs_tensor = pto.TensorType(rank=2, dtype=mx.lhs)
        rhs_tensor = pto.TensorType(rank=2, dtype=mx.rhs)
        lhs_scale_tensor = pto.TensorType(rank=2, dtype=mx.scale)
        rhs_scale_tensor = pto.TensorType(rank=2, dtype=mx.scale)
        bias_tensor = pto.TensorType(rank=2, dtype=mx.acc)

        lhs_tile_view = pto.SubTensorType(shape=[M, K], dtype=mx.lhs)
        rhs_tile_view = pto.SubTensorType(shape=[K, N], dtype=mx.rhs)
        lhs_scale_tile_view = pto.SubTensorType(shape=[M, scale_k], dtype=mx.scale)
        rhs_scale_tile_view = pto.SubTensorType(shape=[scale_k, N], dtype=mx.scale)
        bias_tile_view = pto.SubTensorType(shape=[1, N], dtype=mx.acc)

        lhs_tile = pto.TileBufType(shape=[M, K], dtype=mx.lhs, memory_space="LEFT")
        rhs_tile = pto.TileBufType(shape=[K, N], dtype=mx.rhs, memory_space="RIGHT")
        lhs_scale_tile = pto.LeftScaleTileBufType(shape=[M, scale_k], dtype=mx.scale)
        rhs_scale_tile = pto.RightScaleTileBufType(shape=[scale_k, N], dtype=mx.scale)
        bias_tile = pto.TileBufType(shape=[1, N], dtype=mx.acc, memory_space="BIAS")
        acc_tile = pto.TileBufType(shape=[M, N], dtype=mx.acc, memory_space="ACC")

        return locals()

    const = pto.const

    @to_ir_module(meta_data=meta_data)
    def matmul_mxfp8(
        a_ptr: "ptr_lhs",
        a_scale_ptr: "ptr_scale",
        b_ptr: "ptr_rhs",
        b_scale_ptr: "ptr_scale",
        bias_ptr: "ptr_bias",
    ) -> None:
        c0 = const(0)
        c1 = const(1)
        cM = const(M)
        cK = const(K)
        cN = const(N)
        cScaleK = const(scale_k)

        tv_a = pto.as_tensor(lhs_tensor, ptr=a_ptr, shape=[cM, cK], strides=[cK, c1])
        tv_b = pto.as_tensor(rhs_tensor, ptr=b_ptr, shape=[cK, cN], strides=[cN, c1])
        tv_scale_a = pto.as_tensor(lhs_scale_tensor, ptr=a_scale_ptr, shape=[cM, cScaleK], strides=[cScaleK, c1])
        tv_scale_b = pto.as_tensor(rhs_scale_tensor, ptr=b_scale_ptr, shape=[cScaleK, cN], strides=[cN, c1])
        tv_bias = pto.as_tensor(bias_tensor, ptr=bias_ptr, shape=[c1, cN], strides=[cN, c1])

        sv_a = pto.slice_view(lhs_tile_view, source=tv_a, offsets=[c0, c0], sizes=[cM, cK])
        sv_b = pto.slice_view(rhs_tile_view, source=tv_b, offsets=[c0, c0], sizes=[cK, cN])
        sv_scale_a = pto.slice_view(lhs_scale_tile_view, source=tv_scale_a, offsets=[c0, c0], sizes=[cM, cScaleK])
        sv_scale_b = pto.slice_view(rhs_scale_tile_view, source=tv_scale_b, offsets=[c0, c0], sizes=[cScaleK, cN])
        sv_bias = pto.slice_view(bias_tile_view, source=tv_bias, offsets=[c0, c0], sizes=[c1, cN])

        with pto.cube_section():
            a_tile = pto.alloc_tile(lhs_tile)
            b_tile = pto.alloc_tile(rhs_tile)
            a_scale_tile = pto.alloc_tile(lhs_scale_tile)
            b_scale_tile = pto.alloc_tile(rhs_scale_tile)
            bias_tile_buf = pto.alloc_tile(bias_tile)
            acc_tile_buf = pto.alloc_tile(acc_tile)

            pto.load(sv_a, a_tile)
            pto.load(sv_b, b_tile)
            pto.load(sv_scale_a, a_scale_tile)
            pto.load(sv_scale_b, b_scale_tile)
            pto.load(sv_bias, bias_tile_buf)
            pto.matmul_mx_bias(a_tile, a_scale_tile, b_tile, b_scale_tile, bias_tile_buf, acc_tile_buf)

    return matmul_mxfp8


if __name__ == "__main__":
    print(build())
