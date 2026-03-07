from mlir.dialects import arith


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


__all__ = ["Value", "_unwrap", "wrap_value"]
