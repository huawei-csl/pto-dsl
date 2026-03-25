from mlir.dialects import arith as _arith
from mlir.dialects import pto as _pto
from mlir.ir import BoolAttr, IntegerType

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


def min(lhs, rhs, out):
    _pto.TMinOp(lhs, rhs, out)


def max(lhs, rhs, out):
    _pto.TMaxOp(lhs, rhs, out)


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


def extract(source, index_row, index_col, out):
    _pto.TExtractOp(
        src=source, indexRow=_unwrap(index_row), indexCol=_unwrap(index_col), dst=out
    )


def row_sum(src, tmp, dst):
    _pto.TRowSumOp(src=src, tmp=tmp, dst=dst)


def row_min(src, tmp, dst):
    _pto.TRowMinOp(src=src, tmp=tmp, dst=dst)


def row_max(src, tmp, dst):
    _pto.TRowMaxOp(src=src, tmp=tmp, dst=dst)


def row_prod(src, tmp, dst):
    _pto.TRowProdOp(src=src, tmp=tmp, dst=dst)


def row_expand(src, dst):
    _pto.TRowExpandOp(src=src, dst=dst)


def row_expand_sub(src0, src1, dst):
    _pto.TRowExpandSubOp(src0=src0, src1=src1, dst=dst)


def row_expand_div(src0, src1, dst):
    _pto.TRowExpandDivOp(src0=src0, src1=src1, dst=dst)


def row_expand_mul(src0, src1, dst):
    _pto.TRowExpandMulOp(src0=src0, src1=src1, dst=dst)


def col_sum(src, tmp, dst, is_binary=True):
    _pto.TColSumOp(src=src, dst=dst, tmp=tmp, isBinary=BoolAttr.get(is_binary))


def col_min(src, dst):
    _pto.TColMinOp(src=src, dst=dst)


def col_max(src, dst):
    _pto.TColMaxOp(src=src, dst=dst)


def col_prod(src, tmp, dst, is_binary=True):
    _pto.TColProdOp(src=src, dst=dst, tmp=tmp, isBinary=BoolAttr.get(is_binary))


def col_expand(src, dst):
    _pto.TColExpandOp(src=src, dst=dst)


def mrgsort(src, dst, block_len):
    i32 = IntegerType.get_signless(32)
    block_len_i32 = _arith.IndexCastOp(i32, _unwrap(block_len)).result
    _pto.TMrgSortOp(srcs=[src], dsts=[dst], blockLen=block_len_i32)


def sort32(src, dst, idx):
    """TSORT32: sort src tile within 32-element blocks, writing interleaved
    (score, index) pairs to dst. idx is an input tile of uint32 indices
    attached to each src element. For float16 src, dst must have 4x the
    columns of src (each element expands to 4 float16 words)."""
    _pto.TSort32Op(src, dst, idx)


def subset(source, offsets, sizes):
    offset_vals = [_unwrap(v) for v in offsets]
    return _pto.subset(source, offset_vals, sizes)


def print(source):
    _pto.tprint(source)


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
    "extract",
    "row_sum",
    "row_min",
    "row_max",
    "row_prod",
    "row_expand",
    "row_expand_sub",
    "row_expand_div",
    "row_expand_mul",
    "col_sum",
    "col_min",
    "col_max",
    "col_prod",
    "col_expand",
    "mrgsort",
    "sort32",
    "subset",
]
