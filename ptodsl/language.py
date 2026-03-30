from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

from mlir import ir as mlir_ir
from mlir.dialects import arith, pto, scf
from mlir.ir import F16Type, F32Type, IndexType, InsertionPoint, IntegerType

from ._constexpr import (
    Constexpr,
    const_expr,
    range_constexpr,
    require_static_int,
    require_static_int_sequence,
)
from .api import micro as _micro_api
from .api._micro_registry import MICRO_OPS
from .api.scalar import LazyTypeAlias, resolve_type


def _unwrap(value):
    if isinstance(value, Value):
        return value.raw
    return value


class Value:
    # TODO: generalize to more comprehensive wrappers like https://github.com/makslevental/mlir-python-extras/blob/0.0.8.2/mlir/extras/dialects/ext/arith.py
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


@dataclass(frozen=True)
class MXFP8DType:
    lhs: object
    rhs: object
    scale: object
    acc: object
    scale_factor: int = 32

    @property
    def data(self):
        return self.lhs

    def scale_k(self, k):
        if k % self.scale_factor != 0:
            raise ValueError(f"k={k} must be divisible by scale_factor={self.scale_factor} for MXFP8.")
        return k // self.scale_factor


def _get_mlir_float_type(alias_name, *type_names):
    def _resolve():
        for type_name in type_names:
            type_ctor = getattr(mlir_ir, type_name, None)
            if type_ctor is not None:
                return type_ctor.get()
        supported = ", ".join(type_names)
        raise AttributeError(
            f"module '{__name__}' has no attribute '{alias_name}' because the active MLIR "
            f"Python bindings do not expose any of: {supported}"
        )

    return LazyTypeAlias(alias_name, _resolve)


def make_mxfp8(*, lhs="e5m2", rhs="e5m2", acc=None, scale_factor=32):
    variants = {
        "e4m3": __getattr__("fp8_e4m3"),
        "e5m2": __getattr__("fp8_e5m2"),
    }
    if lhs not in variants:
        raise ValueError(f"Unsupported lhs variant '{lhs}'. Expected one of: {', '.join(sorted(variants))}.")
    if rhs not in variants:
        raise ValueError(f"Unsupported rhs variant '{rhs}'. Expected one of: {', '.join(sorted(variants))}.")
    return MXFP8DType(
        lhs=variants[lhs],
        rhs=variants[rhs],
        scale=__getattr__("fp8_e8m0"),
        acc=__getattr__("float32") if acc is None else acc,
        scale_factor=scale_factor,
    )


def __getattr__(name):
    # Keep aliases conservative and only expose types that map cleanly to MLIR/PTO.
    if name == "bool":
        return LazyTypeAlias(name, lambda: IntegerType.get_signless(1))
    if name == "float32":
        return LazyTypeAlias(name, lambda: F32Type.get())
    if name == "float16":
        return LazyTypeAlias(name, lambda: F16Type.get())
    if name == "bfloat16":
        return _get_mlir_float_type(name, "BF16Type")
    if name in ("fp8_e4m3", "float8_e4m3"):
        return _get_mlir_float_type(name, "Float8E4M3FNType", "Float8E4M3FNUZType")
    if name in ("fp8_e5m2", "float8_e5m2"):
        return _get_mlir_float_type(name, "Float8E5M2Type", "Float8E5M2FNUZType")
    if name in ("fp8_e8m0", "float8_e8m0"):
        return _get_mlir_float_type(name, "Float8E8M0FNUType", "Float8E8M0FNType")
    if name == "mxfp8":
        return make_mxfp8(lhs="e5m2", rhs="e5m2")
    if name == "mxfp8_e4m3":
        return make_mxfp8(lhs="e4m3", rhs="e4m3")
    if name == "mxfp8_e5m2":
        return make_mxfp8(lhs="e5m2", rhs="e5m2")
    if name == "int32":
        return LazyTypeAlias(name, lambda: IntegerType.get_signless(32))
    if name == "int16":
        return LazyTypeAlias(name, lambda: IntegerType.get_signless(16))
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _resolve_memory_space(memory_space):
    if memory_space is None:
        return None
    if isinstance(memory_space, str):
        return getattr(pto.AddressSpace, memory_space)
    if hasattr(memory_space, "value") and not isinstance(memory_space, int):
        return int(memory_space.value)
    return memory_space


def PtrType(dtype, memory_space=None):
    resolved = resolve_type(dtype)
    memory_space = _resolve_memory_space(memory_space)
    if memory_space is None:
        return pto.PtrType.get(resolved)
    return pto.PtrType.get(resolved, memory_space)


def VRegType(lanes, dtype):
    return pto.VRegType.get(
        require_static_int(lanes, context="VRegType.lanes"),
        resolve_type(dtype),
    )


def MaskType():
    return pto.MaskType.get()


def AlignType():
    return pto.AlignType.get()


def TensorType(*, rank, dtype):
    return pto.TensorViewType.get(
        require_static_int(rank, context="TensorType.rank"),
        resolve_type(dtype),
    )


def SubTensorType(*, shape, dtype):
    return pto.PartitionTensorViewType.get(
        require_static_int_sequence(shape, context="SubTensorType.shape"),
        resolve_type(dtype),
    )


class TileBufConfig:
    def __init__(self, blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null"):
        # TODO: expose and validate a broader set of tile buffer knobs if PTO adds
        # more layout/padding/fractal settings that should be configurable here.
        self._bl = pto.BLayoutAttr.get(getattr(pto.BLayout, blayout))
        self._sl = pto.SLayoutAttr.get(getattr(pto.SLayout, slayout))
        self._pd = pto.PadValueAttr.get(getattr(pto.PadValue, pad))
        self._s_fractal_size = require_static_int(
            s_fractal_size, context="TileBufConfig.s_fractal_size"
        )

    @property
    def attr(self):
        return pto.TileBufConfigAttr.get(self._bl, self._sl, self._s_fractal_size, self._pd)


def _default_tile_config(memory_space, shape):
    space = memory_space.upper()
    # Defaults mirror the explicit configs used by the verbose matmul builder.
    if space == "MAT":
        if len(shape) >= 1 and shape[0] == 1:
            return TileBufConfig(blayout="RowMajor", slayout="NoneBox", s_fractal_size=pto.TileConfig.fractalABSize)
        return TileBufConfig(blayout="ColMajor", slayout="RowMajor", s_fractal_size=pto.TileConfig.fractalABSize)
    if space == "LEFT":
        return TileBufConfig(blayout="RowMajor", slayout="RowMajor", s_fractal_size=pto.TileConfig.fractalABSize)
    if space == "RIGHT":
        return TileBufConfig(blayout="RowMajor", slayout="ColMajor", s_fractal_size=pto.TileConfig.fractalABSize)
    if space == "ACC":
        return TileBufConfig(blayout="ColMajor", slayout="RowMajor", s_fractal_size=pto.TileConfig.fractalCSize)
    if space == "BIAS":
        return TileBufConfig(blayout="RowMajor", slayout="NoneBox", s_fractal_size=pto.TileConfig.fractalABSize)
    if space == "SCALING":
        return TileBufConfig(blayout="RowMajor", slayout="NoneBox", s_fractal_size=pto.TileConfig.fractalABSize)
    if space == "VEC":
        return TileBufConfig()
    raise ValueError(f"Unsupported memory_space '{memory_space}' for default tile config.")


def TileBufType(*, shape, dtype, memory_space, valid_shape=None, config=None):
    shape = require_static_int_sequence(shape, context="TileBufType.shape")
    space = pto.AddressSpaceAttr.get(getattr(pto.AddressSpace, memory_space))
    if valid_shape is None:
        valid_shape = shape
    else:
        valid_shape = require_static_int_sequence(
            valid_shape, context="TileBufType.valid_shape"
        )
    if config is None:
        config = _default_tile_config(memory_space, shape)
    cfg = config.attr if isinstance(config, TileBufConfig) else config
    return pto.TileBufType.get(shape, resolve_type(dtype), space, valid_shape, cfg)


def LeftScaleTileBufType(*, shape, dtype, valid_shape=None, config=None):
    if config is None:
        config = TileBufConfig(
            blayout="RowMajor",
            slayout="RowMajor",
            s_fractal_size=pto.TileConfig.fractalMxSize,
        )
    return TileBufType(shape=shape, dtype=dtype, memory_space="SCALING", valid_shape=valid_shape, config=config)


def RightScaleTileBufType(*, shape, dtype, valid_shape=None, config=None):
    if config is None:
        config = TileBufConfig(
            blayout="ColMajor",
            slayout="ColMajor",
            s_fractal_size=pto.TileConfig.fractalMxSize,
        )
    return TileBufType(shape=shape, dtype=dtype, memory_space="SCALING", valid_shape=valid_shape, config=config)


def const(value):
    return Value(arith.ConstantOp(IndexType.get(), value).result)


def get_block_idx():
    return Value(pto.GetBlockIdxOp().result)


def get_subblock_idx():
    return Value(pto.GetSubBlockIdxOp().result)


def get_subblock_num():
    return Value(pto.GetSubBlockNumOp().result)


def get_block_num():
    return Value(pto.GetBlockNumOp().result)


def index_cast(value, index_type=IndexType):
    if hasattr(index_type, "get"):
        dst = index_type.get()
    else:
        dst = index_type
    return Value(arith.IndexCastOp(dst, _unwrap(value)).result)


def as_tensor(tensor_type, *, ptr, shape, strides):
    shape_vals = [_unwrap(v) for v in shape]
    stride_vals = [_unwrap(v) for v in strides]
    return pto.MakeTensorViewOp(tensor_type, _unwrap(ptr), shape_vals, stride_vals).result


def slice_view(subtensor_type, *, source, offsets, sizes):
    offset_vals = [_unwrap(v) for v in offsets]
    size_vals = [_unwrap(v) for v in sizes]
    return pto.PartitionViewOp(subtensor_type, source, offsets=offset_vals, sizes=size_vals).result


@contextmanager
def vector_section():
    section = pto.SectionVectorOp()
    block = section.body.blocks.append()
    with InsertionPoint(block):
        yield


@contextmanager
def cube_section():
    section = pto.SectionCubeOp()
    block = section.body.blocks.append()
    with InsertionPoint(block):
        yield


def for_range(start, stop, step):
    loop = scf.ForOp(_unwrap(start), _unwrap(stop), _unwrap(step))
    with InsertionPoint(loop.body):
        yield Value(loop.induction_variable)
        scf.YieldOp([])


def alloc_tile(tile_type, *, valid_row=None, valid_col=None):
    kwargs = {}
    if valid_row is not None:
        kwargs["valid_row"] = _unwrap(valid_row)
    if valid_col is not None:
        kwargs["valid_col"] = _unwrap(valid_col)
    return pto.AllocTileOp(tile_type, **kwargs).result


def subset(source, offsets, sizes):
    offset_vals = [_unwrap(v) for v in offsets]
    return pto.subset(source, offset_vals, sizes)


def load(source, dest):
    pto.TLoadOp(None, source, dest)


def mov(source, dest):
    pto.TMovOp(None, source, dest)


def add(lhs, rhs, out):
    pto.TAddOp(lhs, rhs, out)


def sub(lhs, rhs, out):
    pto.TSubOp(lhs, rhs, out)


def div(lhs, rhs, out):
    pto.TDivOp(lhs, rhs, out)


def mul(lhs, rhs, out):
    pto.TMulOp(lhs, rhs, out)


def or_(lhs, rhs, out):
    pto.TOrOp(lhs, rhs, out)


def gather(src, out, indices=None, *, mask_pattern=None):
    if mask_pattern is not None:
        mp = pto.MaskPatternAttr.get(getattr(pto.MaskPattern, mask_pattern))
        pto.TGatherOp(src, out, maskPattern=mp)
    else:
        pto.TGatherOp(src, out, indices=indices)


def exp(inp, out):
    pto.TExpOp(inp, out)


def log(inp, out):
    pto.TLogOp(inp, out)


def relu(inp, out):
    pto.TReluOp(inp, out)


def abs(inp, out):
    pto.TAbsOp(inp, out)


def sqrt(inp, out):
    pto.TSqrtOp(inp, out)


def store(source, dest):
    pto.TStoreOp(None, source, dest)


def matmul(lhs, rhs, out):
    pto.TMatmulOp(None, lhs, rhs, out)


def matmul_bias(lhs, rhs, bias, out):
    pto.TMatmulBiasOp(None, lhs, rhs, bias, out)


def matmul_acc(acc, lhs, rhs, out):
    pto.TMatmulAccOp(None, acc, lhs, rhs, out)


def _emit_dps_op(op_name, *operands):
    op_ctor = getattr(pto, op_name, None)
    if op_ctor is not None:
        return op_ctor(None, *operands)
    generic_name = {
        "TMatmulMxOp": "pto.tmatmul.mx",
        "TMatmulMxAccOp": "pto.tmatmul.mx.acc",
        "TMatmulMxBiasOp": "pto.tmatmul.mx.bias",
    }[op_name]
    return mlir_ir.Operation.create(generic_name, operands=list(operands))


def matmul_mx(lhs, lhs_scale, rhs, rhs_scale, out):
    _emit_dps_op("TMatmulMxOp", lhs, lhs_scale, rhs, rhs_scale, out)


def matmul_mx_acc(acc, lhs, lhs_scale, rhs, rhs_scale, out):
    _emit_dps_op("TMatmulMxAccOp", acc, lhs, lhs_scale, rhs, rhs_scale, out)


def matmul_mx_bias(lhs, lhs_scale, rhs, rhs_scale, bias, out):
    _emit_dps_op("TMatmulMxBiasOp", lhs, lhs_scale, rhs, rhs_scale, bias, out)


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


class _IfElseBranch:
    def __init__(self, if_op):
        self._if_op = if_op
    @contextmanager
    def else_context(self):
        with InsertionPoint(self._if_op.else_block):
            yield
            scf.YieldOp([])

@contextmanager
def if_context(condition, has_else=False):
    if has_else:
        op = scf.IfOp(_unwrap(condition), [], hasElse=True)
        branch = _IfElseBranch(op)
    else:
        op = scf.IfOp(_unwrap(condition))
        branch = None

    with InsertionPoint(op.then_block):
        yield branch
        scf.YieldOp([])


def cond(condition, then_builder, else_builder):
    op = scf.IfOp(_unwrap(condition), [], hasElse=True)
    with InsertionPoint(op.then_block):
        then_builder()
        scf.YieldOp([])
    with InsertionPoint(op.else_block):
        else_builder()
        scf.YieldOp([])
    return op

def _resolve_sync_op(sync_op):
    if isinstance(sync_op, str):
        normalized = sync_op.strip().upper()
        if not normalized.startswith("T"):
            normalized = f"T{normalized}"
        try:
            return getattr(pto, normalized)
        except AttributeError as exc:
            raise ValueError(f"Unsupported sync op type '{sync_op}'.") from exc
    return sync_op


def _resolve_event_id(event_id):
    if isinstance(event_id, int):
        if event_id < 0 or event_id > 7:
            raise ValueError(f"event_id must be in range [0, 7], got {event_id}.")
        return getattr(pto, f"EVENT_ID{event_id}")
    return event_id


def record_event(record_op, wait_op, event_id: int|Sequence[int]=0):
    if not isinstance(event_id, int):
        for eid in event_id:
            pto.record_event(_resolve_sync_op(record_op), _resolve_sync_op(wait_op), _resolve_event_id(eid))
    else:
        pto.record_event(_resolve_sync_op(record_op), _resolve_sync_op(wait_op), _resolve_event_id(event_id))



def wait_event(record_op, wait_op, event_id: int|Sequence[int]=0):
    if not isinstance(event_id, int):
        for eid in event_id:
            pto.wait_event(_resolve_sync_op(record_op), _resolve_sync_op(wait_op), _resolve_event_id(eid))
    else:
        pto.wait_event(_resolve_sync_op(record_op), _resolve_sync_op(wait_op), _resolve_event_id(event_id))


def record_wait_pair(record_op, wait_op, event_id: int|Sequence[int]=0):
    rec = _resolve_sync_op(record_op)
    w = _resolve_sync_op(wait_op)
    ev = _resolve_event_id(event_id)
    pto.record_event(rec, w, ev)
    pto.wait_event(rec, w, ev)


def barrier(sync_op):
    pto.barrier(_resolve_sync_op(sync_op))


def barrier_sync(sync_op):
    pto.barrier_sync(pto.SyncOpTypeAttr.get(_resolve_sync_op(sync_op)))


def row_sum(src, tmp, dst):
    pto.TRowSumOp(src = src, tmp = tmp, dst = dst)


for _name in MICRO_OPS:
    globals()[_name] = getattr(_micro_api, _name)
