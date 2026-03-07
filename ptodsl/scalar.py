from mlir.dialects import arith
from mlir.ir import F16Type, F32Type, IndexType, IntegerType

from .core import Value, _unwrap


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


def const(value):
    return Value(arith.ConstantOp(IndexType.get(), value).result)


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
