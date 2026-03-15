from ptodsl import pto, to_ir_module

from tools.ptodsl_capability_gate import build_report


def control_flow_meta():
    return {
        "i32": pto.int32,
    }


def control_flow_kernel(n: "i32") -> "i32":
    total = pto.const(0, dtype=pto.int32)
    one = pto.const(1, dtype=pto.int32)
    zero = pto.const(0, dtype=pto.int32)
    four = pto.const(4, dtype=pto.int32)

    if n > total:
        total = n
    else:
        total = total + one

    i = zero
    while i < n:
        total = total + one
        i = i + one

    for _ in range(n):
        total = total + one

    return total + (four - four)


def unsupported_meta():
    return {
        "i32": pto.int32,
    }


def unsupported_kernel(n: "i32") -> None:
    values = [n for _ in range(2)]
    return values


def fillpad_meta():
    dtype = pto.float16
    ptr = pto.PtrType(dtype)
    tensor = pto.TensorType(rank=2, dtype=dtype)
    sub_src = pto.SubTensorType(shape=[1, 32], dtype=dtype)
    sub_dst = pto.SubTensorType(shape=[1, 32], dtype=dtype)
    cfg = pto.TileConfig()
    src_tile = pto.TileType(shape=[1, 32], valid_shape=[1, 16], dtype=dtype, memory_space="VEC", config=cfg)
    dst_tile = pto.TileType(shape=[1, 32], valid_shape=[1, 16], dtype=dtype, memory_space="VEC", config=cfg)
    return {
        "ptr": ptr,
        "tensor": tensor,
        "sub_src": sub_src,
        "sub_dst": sub_dst,
        "src_tile": src_tile,
        "dst_tile": dst_tile,
    }


def fillpad_kernel(src_ptr: "ptr", dst_ptr: "ptr") -> None:
    c0 = pto.const(0)
    c1 = pto.const(1)
    c32 = pto.const(32)
    src = pto.as_tensor(tensor, ptr=src_ptr, shape=[c1, c32], strides=[c32, c1])
    dst = pto.as_tensor(tensor, ptr=dst_ptr, shape=[c1, c32], strides=[c32, c1])

    with pto.section.vector():
        src_tb = pto.alloc_tile(src_tile)
        dst_tb = pto.alloc_tile(dst_tile)
        pto.load(
            pto.slice_view(sub_src, source=src, offsets=[c0, c0], sizes=[c1, c32]),
            src_tb,
        )
        pto.fillpad(src_tb, dst_tb)
        pto.store(dst_tb, pto.slice_view(sub_dst, source=dst, offsets=[c0, c0], sizes=[c1, c32]))


def test_python_if_for_while_lower_to_scf():
    module = to_ir_module(meta_data=control_flow_meta)(control_flow_kernel)
    ir = str(module)
    assert "scf.if" in ir
    assert "scf.for" in ir
    assert "scf.while" in ir
    assert "return " in ir


def test_fillpad_is_emitted_from_ptodsl():
    module = to_ir_module(meta_data=fillpad_meta)(fillpad_kernel)
    assert "pto.tfillpad" in str(module)


def test_static_python_range_is_unrolled():
    def static_meta():
        return {"i32": pto.int32}

    def static_kernel() -> "i32":
        total = pto.const(0, dtype=pto.int32)
        for i in range(4):
            total = total + i
        return total

    module = to_ir_module(meta_data=static_meta)(static_kernel)
    ir = str(module)
    assert "scf.for" not in ir
    assert ir.count("arith.constant 0") >= 1
    assert "arith.constant 3" in ir


def test_unsupported_python_constructs_raise_targeted_error():
    try:
        to_ir_module(meta_data=unsupported_meta)(unsupported_kernel)
    except ValueError as exc:
        assert "Unsupported" in str(exc) or "supported" in str(exc)
    else:
        raise AssertionError("unsupported Python construct should fail AST lowering")


def test_capability_gate_report_is_green():
    report = build_report()
    assert report["ok"], report


def test_closure_captured_constants_lower_cleanly():
    hidden = 8

    def closure_meta():
        return {"i32": pto.int32}

    def closure_kernel(n: "i32") -> "i32":
        return n + pto.const(hidden, dtype=pto.int32)

    module = to_ir_module(meta_data=closure_meta)(closure_kernel)
    assert "arith.constant 8" in str(module)


def test_pto_scalar_compare_helpers_are_exported():
    lhs = pto.const(3, dtype=pto.int32)
    rhs = pto.const(2, dtype=pto.int32)
    assert "arith.cmpi sgt" in str(pto.gt(lhs, rhs).owner)


def test_if_without_else_can_still_carry_values():
    def carry_meta():
        return {"i32": pto.int32}

    def carry_kernel(n: "i32") -> "i32":
        total = pto.const(0, dtype=pto.int32)
        if pto.gt(n, total):
            total = n
        return total

    module = to_ir_module(meta_data=carry_meta)(carry_kernel)
    ir = str(module)
    assert "scf.if" in ir
    assert "else" in ir
