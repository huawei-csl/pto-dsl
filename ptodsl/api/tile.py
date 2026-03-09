from mlir.dialects import pto as _pto

from .scalar import _unwrap


def mov(source, dest):
    _pto.TMovOp(None, source, dest)


def add(lhs, rhs, out):
    _pto.TAddOp(lhs, rhs, out)


def sub(lhs, rhs, out):
    _pto.TSubOp(lhs, rhs, out)


def div(lhs, rhs, out):
    _pto.TDivOp(lhs, rhs, out)


def mul(lhs, rhs, out):
    _pto.TMulOp(lhs, rhs, out)


def or_(lhs, rhs, out):
    _pto.TOrOp(lhs, rhs, out)


def gather(src, out, indices=None, *, mask_pattern=None):
    if mask_pattern is not None:
        mask = _pto.MaskPatternAttr.get(getattr(_pto.MaskPattern, mask_pattern))
        _pto.TGatherOp(src, out, maskPattern=mask)
    else:
        _pto.TGatherOp(src, out, indices=indices)


def exp(inp, out):
    _pto.TExpOp(inp, out)


def log(inp, out):
    _pto.TLogOp(inp, out)


def relu(inp, out):
    _pto.TReluOp(inp, out)


def abs(inp, out):
    _pto.TAbsOp(inp, out)


def sqrt(inp, out):
    _pto.TSqrtOp(inp, out)


def rsqrt(inp, out):
    _pto.TRsqrtOp(inp, out)


def reciprocal(inp, out):
    _pto.TRecipOp(inp, out)


def matmul(lhs, rhs, out):
    _pto.TMatmulOp(None, lhs, rhs, out)


def matmul_bias(lhs, rhs, bias, out):
    _pto.TMatmulBiasOp(None, lhs, rhs, bias, out)


def matmul_acc(acc, lhs, rhs, out):
    _pto.TMatmulAccOp(None, acc, lhs, rhs, out)


def row_sum(src, tmp, dst):
    _pto.TRowSumOp(src=src, tmp=tmp, dst=dst)


def subset(source, offsets, sizes):
    offset_vals = [_unwrap(v) for v in offsets]
    return _pto.subset(source, offset_vals, sizes)


__all__ = [
    "mov",
    "add",
    "sub",
    "div",
    "mul",
    "or_",
    "gather",
    "exp",
    "log",
    "relu",
    "abs",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "matmul",
    "matmul_bias",
    "matmul_acc",
    "row_sum",
    "subset",
]
