from mlir.dialects import pto as _raw_pto

from .control_flow import cond, if_context, range
from .pto_general import (
    alloc_tile,
    as_tensor,
    cube_section,
    get_block_idx,
    get_block_num,
    get_subblock_idx,
    get_subblock_num,
    load,
    print,
    section,
    slice_view,
    store,
    vector_section,
)
from .scalar import (
    Value,
    ceil_div,
    const,
    div_s,
    index_cast,
    min_u,
    rem_s,
    select,
    wrap_value,
)
from .synchronization import barrier
from .tile import (
    abs,
    add,
    adds,
    col_expand,
    col_max,
    col_min,
    col_prod,
    col_sum,
    cvt,
    div,
    exp,
    expands,
    extract,
    gather,
    log,
    matmul,
    matmul_acc,
    matmul_bias,
    max,
    min,
    mov,
    mul,
    muls,
    or_,
    reciprocal,
    relu,
    row_expand,
    row_max,
    row_min,
    row_prod,
    row_sum,
    rsqrt,
    scatter,
    sqrt,
    sub,
    subset,
    subs,
)
from .type_def import (
    PtrType,
    SubTensorType,
    TensorType,
    TileConfig,
    TileType,
    __getattr__ as _type_getattr,
)


_HIDDEN_RAW_EXPORTS = {"TileBufType"}


def __getattr__(name):
    if name in _HIDDEN_RAW_EXPORTS:
        raise AttributeError(
            f"module '{__name__}' has no attribute '{name}'. Use 'TileType' as the canonical PTODSL tile type."
        )
    try:
        return _type_getattr(name)
    except AttributeError:
        pass
    if hasattr(_raw_pto, name):
        return getattr(_raw_pto, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def fillpad(src, dst):
    return _raw_pto.TFillPadOp(src, dst)


def fillpad_expand(src, dst):
    op_ctor = getattr(_raw_pto, "TFillPadExpandOp", None)
    if op_ctor is None:
        raise AttributeError("PTOAS python bindings do not expose TFillPadExpandOp in this build.")
    return op_ctor(src, dst)


_PUBLIC_EXPORTS = [
    "Value",
    "wrap_value",
    "const",
    "index_cast",
    "ceil_div",
    "div_s",
    "rem_s",
    "min_u",
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
    "PtrType",
    "TensorType",
    "SubTensorType",
    "TileConfig",
    "TileType",
    "get_block_idx",
    "get_subblock_idx",
    "get_subblock_num",
    "get_block_num",
    "as_tensor",
    "slice_view",
    "section",
    "vector_section",
    "cube_section",
    "range",
    "if_context",
    "cond",
    "alloc_tile",
    "load",
    "store",
    "print",
    "barrier",
    "fillpad",
    "fillpad_expand",
    "mov",
    "add",
    "adds",
    "sub",
    "subs",
    "div",
    "mul",
    "muls",
    "or_",
    "min",
    "max",
    "gather",
    "scatter",
    "exp",
    "log",
    "relu",
    "abs",
    "sqrt",
    "rsqrt",
    "reciprocal",
    "cvt",
    "matmul",
    "matmul_bias",
    "matmul_acc",
    "extract",
    "row_sum",
    "row_min",
    "row_max",
    "row_prod",
    "row_expand",
    "col_sum",
    "col_min",
    "col_max",
    "col_prod",
    "col_expand",
    "expands",
    "subset",
]

_RAW_EXPORTS = [
    name for name in dir(_raw_pto)
    if not name.startswith("_") and name not in _HIDDEN_RAW_EXPORTS
]

__all__ = sorted(set(_PUBLIC_EXPORTS + _RAW_EXPORTS))
