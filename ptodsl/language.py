from contextlib import contextmanager

from mlir.dialects import arith, pto
from mlir.ir import F32Type, IndexType, InsertionPoint, IntegerType


def _unwrap(value):
    if isinstance(value, Value):
        return value.raw
    return value


class Value:
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

    def __getattr__(self, item):
        return getattr(self.raw, item)


def wrap_value(value):
    if isinstance(value, Value):
        return value
    return Value(value)


def __getattr__(name):
    # TODO: add more builtin dtype aliases (for example float16/bfloat16/int8/int64)
    # when they are validated against PTO type support.
    if name == "float32":
        return F32Type.get()
    if name == "int32":
        return IntegerType.get_signless(32)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def PtrType(dtype):
    return pto.PtrType.get(dtype)


def TensorType(*, rank, dtype):
    return pto.TensorViewType.get(rank, dtype)


def SubTensorType(*, shape, dtype):
    return pto.PartitionTensorViewType.get(shape, dtype)


class TileBufConfig:
    def __init__(self, blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null"):
        # TODO: expose and validate a broader set of tile buffer knobs if PTO adds
        # more layout/padding/fractal settings that should be configurable here.
        self._bl = pto.BLayoutAttr.get(getattr(pto.BLayout, blayout))
        self._sl = pto.SLayoutAttr.get(getattr(pto.SLayout, slayout))
        self._pd = pto.PadValueAttr.get(getattr(pto.PadValue, pad))
        self._s_fractal_size = s_fractal_size

    @property
    def attr(self):
        return pto.TileBufConfigAttr.get(self._bl, self._sl, self._s_fractal_size, self._pd)


def TileBufType(*, shape, valid_shape, dtype, memory_space, config):
    space = pto.AddressSpaceAttr.get(getattr(pto.AddressSpace, memory_space))
    cfg = config.attr if isinstance(config, TileBufConfig) else config
    return pto.TileBufType.get(shape, dtype, space, valid_shape, cfg)


def const(value):
    return Value(arith.ConstantOp(IndexType.get(), value).result)


def get_block_idx():
    return Value(pto.GetBlockIdxOp().result)


def get_subblock_idx():
    return Value(pto.GetSubBlockIdxOp().result)


def get_subblock_num():
    return Value(pto.GetSubBlockNumOp().result)


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


def alloc_tile(tile_type, *, valid_row, valid_col):
    return pto.AllocTileOp(tile_type, valid_row=_unwrap(valid_row), valid_col=_unwrap(valid_col)).result


def load(source, dest):
    pto.TLoadOp(None, source, dest)


def add(lhs, rhs, out):
    pto.TAddOp(lhs, rhs, out)


def store(source, dest):
    pto.TStoreOp(None, source, dest)
