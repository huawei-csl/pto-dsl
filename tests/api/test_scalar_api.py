from mlir.dialects import arith
from mlir.ir import IntegerType

from ptodsl import scalar


class _Box:
    def __init__(self, raw):
        self.raw = raw


def _i32_const(value):
    return scalar.Value(arith.ConstantOp(IntegerType.get_signless(32), value).result)


def test_scalar_dtype_aliases_resolve_inside_mlir_context(mlir_ctx):
    assert repr(scalar.float32) == "float32"
    assert str(scalar.resolve_type(scalar.bool)) == "i1"
    assert str(scalar.resolve_type(scalar.float16)) == "f16"
    assert str(scalar.resolve_type(scalar.float32)) == "f32"
    assert str(scalar.resolve_type(scalar.int16)) == "i16"
    assert str(scalar.resolve_type(scalar.int32)) == "i32"
    assert str(scalar.resolve_type(scalar.uint32)) == "ui32"


def test_wrap_value_and_unwrap_handle_nested_raw_wrappers():
    value = scalar.Value("inner")

    assert scalar.wrap_value(value) is value
    assert isinstance(scalar.wrap_value("raw"), scalar.Value)
    assert scalar._unwrap(_Box(_Box(value))) == "inner"


def test_scalar_arithmetic_and_helper_builders_emit_expected_arith_ops(mlir_ctx):
    lhs = _i32_const(8)
    rhs = _i32_const(2)

    assert "arith.addi" in str((lhs + rhs).raw.owner)
    assert "arith.subi" in str((lhs - rhs).raw.owner)
    assert "arith.muli" in str((lhs * rhs).raw.owner)
    assert "arith.divsi" in str((lhs // rhs).raw.owner)
    assert "arith.remsi" in str((lhs % rhs).raw.owner)
    assert "arith.cmpi" in str((lhs < rhs).raw.owner)
    assert "arith.index_cast" in str(scalar.index_cast(lhs).raw.owner)
    assert "arith.ceildivsi" in str(scalar.ceil_div(lhs, rhs).raw.owner)
    assert "arith.minui" in str(scalar.min_u(lhs, rhs).raw.owner)
    assert "arith.select" in str(
        scalar.select(scalar.eq(lhs, rhs), lhs, rhs).raw.owner
    )
