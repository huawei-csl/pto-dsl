from mlir.dialects import arith
from mlir.ir import F16Type, F32Type, IndexType, IntegerType


def _unwrap(value):
    if isinstance(value, Value):
        return value.raw
    return value


class Value:
    # TODO: generalize to more comprehensive wrappers like
    # https://github.com/makslevental/mlir-python-extras/blob/0.0.8.2/mlir/extras/dialects/ext/arith.py
    def __init__(self, raw):
        self.raw = raw

    def __mul__(self, other):
        return Value(arith.MulIOp(_unwrap(self), _unwrap(other)).result)

    def __rmul__(self, other):
        return Value(arith.MulIOp(_unwrap(other), _unwrap(self)).result)

    def __add__(self, other):
        return Value(arith.AddIOp(_unwrap(self), _unwrap(other)).result)

    def __radd__(self, other):
        return Value(arith.AddIOp(_unwrap(other), _unwrap(self)).result)

    def __sub__(self, other):
        return Value(arith.SubIOp(_unwrap(self), _unwrap(other)).result)

    def __rsub__(self, other):
        return Value(arith.SubIOp(_unwrap(other), _unwrap(self)).result)

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

    @staticmethod
    def _cmp(lhs, rhs, predicate):
        return Value(arith.CmpIOp(predicate, _unwrap(lhs), _unwrap(rhs)).result)

    def __lt__(self, other):
        return Value._cmp(self, other, arith.CmpIPredicate.slt)

    def __gt__(self, other):
        return Value._cmp(self, other, arith.CmpIPredicate.sgt)

    def __le__(self, other):
        return Value._cmp(self, other, arith.CmpIPredicate.sle)

    def __ge__(self, other):
        return Value._cmp(self, other, arith.CmpIPredicate.sge)

    def __eq__(self, other):
        return Value._cmp(self, other, arith.CmpIPredicate.eq)

    def __ne__(self, other):
        return Value._cmp(self, other, arith.CmpIPredicate.ne)

    def __getattr__(self, item):
        return getattr(self.raw, item)


def wrap_value(value):
    if isinstance(value, Value):
        return value
    return Value(value)


def __getattr__(name):
    # TODO: add more builtin dtype aliases (for example float16/bfloat16/int8/int64)
    # when they are validated against PTO type support.
    if name == "bool":
        return IntegerType.get_signless(1)
    if name == "float32":
        return F32Type.get()
    if name == "float16":
        return F16Type.get()
    if name == "int32":
        return IntegerType.get_signless(32)
    if name == "int16":
        return IntegerType.get_signless(16)
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
    return Value(arith.CmpIOp(arith.CmpIPredicate.eq, _unwrap(a), _unwrap(b)).result)


def lt(a, b):
    return Value(arith.CmpIOp(arith.CmpIPredicate.slt, _unwrap(a), _unwrap(b)).result)


def gt(a, b):
    return Value(arith.CmpIOp(arith.CmpIPredicate.sgt, _unwrap(a), _unwrap(b)).result)


def ge(a, b):
    return Value(arith.CmpIOp(arith.CmpIPredicate.sge, _unwrap(a), _unwrap(b)).result)


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
]
