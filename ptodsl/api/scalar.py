from mlir.dialects import arith
from mlir.ir import IndexType, Type


_DTYPE_SPELLINGS = {
    "bool": "i1",
    "int8": "i8",
    "uint8": "ui8",
    "int16": "i16",
    "uint16": "ui16",
    "int32": "i32",
    "uint32": "ui32",
    "int64": "i64",
    "uint64": "ui64",
    "float16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "float8_e4m3fn": "f8E4M3FN",
    "float8_e5m2": "f8E5M2",
}


def _is_float_type(ty):
    text = str(ty)
    return text.startswith("f") or text == "bf16"


def _coerce_scalar_like(value, reference_type=None):
    if isinstance(value, Value):
        return value.raw
    if reference_type is not None and isinstance(value, (bool, int, float)):
        return const(value, dtype=reference_type).raw
    return value


def _binary_op(lhs, rhs, int_ctor, float_ctor):
    lhs_ref = lhs.type if isinstance(lhs, Value) else None
    rhs_ref = rhs.type if isinstance(rhs, Value) else None
    lhs = _coerce_scalar_like(lhs, rhs_ref)
    rhs = _coerce_scalar_like(rhs, lhs_ref)
    ctor = float_ctor if _is_float_type(lhs.type) else int_ctor
    return Value(ctor(lhs, rhs).result)


def _cmp_op(lhs, rhs, int_predicate, float_predicate):
    lhs_ref = lhs.type if isinstance(lhs, Value) else None
    rhs_ref = rhs.type if isinstance(rhs, Value) else None
    lhs = _coerce_scalar_like(lhs, rhs_ref)
    rhs = _coerce_scalar_like(rhs, lhs_ref)
    if _is_float_type(lhs.type):
        return Value(arith.CmpFOp(float_predicate, lhs, rhs).result)
    return Value(arith.CmpIOp(int_predicate, lhs, rhs).result)


def _unwrap(value):
    if isinstance(value, Value):
        return value.raw
    return value


class Value:
    def __init__(self, raw):
        self.raw = raw

    def __bool__(self):
        raise TypeError(
            "PTODSL values cannot drive Python control flow directly. "
            "Use Python syntax inside @to_ir_module so the AST frontend can lower it to scf."
        )

    def __mul__(self, other):
        return _binary_op(self, other, arith.MulIOp, arith.MulFOp)

    def __rmul__(self, other):
        return _binary_op(other, self, arith.MulIOp, arith.MulFOp)

    def __add__(self, other):
        return _binary_op(self, other, arith.AddIOp, arith.AddFOp)

    def __radd__(self, other):
        return _binary_op(other, self, arith.AddIOp, arith.AddFOp)

    def __sub__(self, other):
        return _binary_op(self, other, arith.SubIOp, arith.SubFOp)

    def __rsub__(self, other):
        return _binary_op(other, self, arith.SubIOp, arith.SubFOp)

    def __floordiv__(self, other):
        return Value(arith.DivSIOp(_unwrap(self), _unwrap(other)).result)

    def __rfloordiv__(self, other):
        return Value(arith.DivSIOp(_unwrap(other), _unwrap(self)).result)

    def __truediv__(self, other):
        return Value(arith.DivFOp(_unwrap(self), _unwrap(other)).result)

    def __rtruediv__(self, other):
        return Value(arith.DivFOp(_unwrap(other), _unwrap(self)).result)

    def __mod__(self, other):
        return Value(arith.RemSIOp(_unwrap(self), _unwrap(other)).result)

    def __rmod__(self, other):
        return Value(arith.RemSIOp(_unwrap(other), _unwrap(self)).result)

    def __lt__(self, other):
        return _cmp_op(self, other, arith.CmpIPredicate.slt, arith.CmpFPredicate.OLT)

    def __gt__(self, other):
        return _cmp_op(self, other, arith.CmpIPredicate.sgt, arith.CmpFPredicate.OGT)

    def __le__(self, other):
        return _cmp_op(self, other, arith.CmpIPredicate.sle, arith.CmpFPredicate.OLE)

    def __ge__(self, other):
        return _cmp_op(self, other, arith.CmpIPredicate.sge, arith.CmpFPredicate.OGE)

    def __eq__(self, other):
        return _cmp_op(self, other, arith.CmpIPredicate.eq, arith.CmpFPredicate.OEQ)

    def __ne__(self, other):
        return _cmp_op(self, other, arith.CmpIPredicate.ne, arith.CmpFPredicate.ONE)

    def __neg__(self):
        zero = const(0.0 if _is_float_type(self.type) else 0, dtype=self.type)
        return zero - self

    def __and__(self, other):
        return Value(arith.AndIOp(_unwrap(self), _unwrap(other)).result)

    def __rand__(self, other):
        return Value(arith.AndIOp(_unwrap(other), _unwrap(self)).result)

    def __or__(self, other):
        return Value(arith.OrIOp(_unwrap(self), _unwrap(other)).result)

    def __ror__(self, other):
        return Value(arith.OrIOp(_unwrap(other), _unwrap(self)).result)

    def __invert__(self):
        one = const(1, dtype=self.type)
        return Value(arith.XOrIOp(_unwrap(self), _unwrap(one)).result)

    def __getattr__(self, item):
        return getattr(self.raw, item)


def wrap_value(value):
    if isinstance(value, Value):
        return value
    return Value(value)


def __getattr__(name):
    if name in _DTYPE_SPELLINGS:
        return Type.parse(_DTYPE_SPELLINGS[name])
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def const(value, dtype=None):
    if dtype is None:
        dtype = IndexType.get()
    return Value(arith.ConstantOp(dtype, value).result)


def index_cast(value, index_type=IndexType):
    if hasattr(index_type, "get"):
        dst = index_type.get()
    else:
        dst = index_type
    return Value(arith.IndexCastOp(dst, _unwrap(value)).result)


def ceil_div(a, b):
    return Value(arith.CeilDivSIOp(_unwrap(a), _unwrap(b)).result)


def div_s(a, b):
    return Value(arith.DivSIOp(_unwrap(a), _unwrap(b)).result)


def rem_s(a, b):
    return Value(arith.RemSIOp(_unwrap(a), _unwrap(b)).result)


def min_u(a, b):
    return Value(arith.MinUIOp(_unwrap(a), _unwrap(b)).result)


def eq(a, b):
    return _cmp_op(a, b, arith.CmpIPredicate.eq, arith.CmpFPredicate.OEQ)


def lt(a, b):
    return _cmp_op(a, b, arith.CmpIPredicate.slt, arith.CmpFPredicate.OLT)


def gt(a, b):
    return _cmp_op(a, b, arith.CmpIPredicate.sgt, arith.CmpFPredicate.OGT)


def ge(a, b):
    return _cmp_op(a, b, arith.CmpIPredicate.sge, arith.CmpFPredicate.OGE)


def select(cond, true_val, false_val):
    return Value(arith.SelectOp(_unwrap(cond), _unwrap(true_val), _unwrap(false_val)).result)


__all__ = [
    "Value",
    "_unwrap",
    "wrap_value",
    "const",
    "index_cast",
    "ceil_div",
    "div_s",
    "rem_s",
    "min_u",
    "eq",
    "lt",
    "gt",
    "ge",
    "select",
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float8_e4m3fn",
    "float8_e5m2",
]
