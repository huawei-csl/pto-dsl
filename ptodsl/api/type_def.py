from mlir.dialects import pto as _pto
from mlir.ir import Context, IntegerType, MemRefType

from . import scalar


def _has_context():
    try:
        return Context.current is not None
    except Exception:
        return False


class _LazyType:
    """Placeholder for an MLIR/PTO type constructed outside an active Context.

    When the type constructors (PtrType, TensorType, etc.) are called without a
    live MLIR context they return a _LazyType instead of raising.  The compiler
    materialises all _LazyType instances found in the kernel function's globals
    and closure once the Context is active.
    """

    def __init__(self, factory, args, kwargs):
        self._factory = factory
        self._args = args
        self._kwargs = kwargs

    def materialize(self):
        return self._factory(*self._args, **self._kwargs)

    def __repr__(self):
        return (
            f"_LazyType({self._factory.__name__}, "
            f"args={self._args!r}, kwargs={self._kwargs!r})"
        )


def _materialize(obj):
    """Materialize obj if it is a _LazyType, otherwise return it unchanged."""
    if isinstance(obj, _LazyType):
        return obj.materialize()
    return obj


_DTYPE_NAMES = {
    "bool",
    "float16",
    "float32",
    "int8",
    "uint8",
    "int16",
    "int32",
    "uint32",
    "int64",
}


def _make_scalar_dtype(name):
    return getattr(scalar, name)


def __getattr__(name):
    # MLIR type factories require an active context, so keep dtype aliases lazy
    # and resolve them only when user code accesses them inside PTO/MLIR setup.
    if name in _DTYPE_NAMES:
        if _has_context():
            return getattr(scalar, name)
        return _LazyType(_make_scalar_dtype, (name,), {})
    if name == "ffts_type":
        return MemRefType.get([256], IntegerType.get_unsigned(64))

    if name.startswith("PIPE_"):
        return _pto.PipeAttr.get(getattr(_pto.PIPE, name))

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _make_ptr_type(dtype):
    return _pto.PtrType.get(_materialize(dtype))


def PtrType(dtype):
    if _has_context():
        return _make_ptr_type(dtype)
    return _LazyType(_make_ptr_type, (dtype,), {})


def _make_tensor_type(rank, shape, dtype):
    dtype = _materialize(dtype)
    if shape is not None:
        if rank is not None and rank != len(shape):
            raise ValueError("TensorType rank must match len(shape).")
        return _pto.TensorViewType.get(shape, dtype)
    if rank is None:
        raise ValueError("TensorType requires either rank or shape.")
    return _pto.TensorViewType.get(rank, dtype)


def TensorType(*, rank=None, shape=None, dtype):
    if _has_context():
        return _make_tensor_type(rank, shape, dtype)
    return _LazyType(_make_tensor_type, (rank, shape, dtype), {})


def _make_sub_tensor_type(shape, dtype):
    return _pto.PartitionTensorViewType.get(shape, _materialize(dtype))


def SubTensorType(*, shape, dtype):
    if _has_context():
        return _make_sub_tensor_type(shape, dtype)
    return _LazyType(_make_sub_tensor_type, (shape, dtype), {})


class TileBufConfig:
    def __init__(
        self, blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null"
    ):
        # TODO: expose and validate a broader set of tile buffer knobs if PTO adds
        # more layout/padding/fractal settings that should be configurable here.
        self._blayout = blayout
        self._slayout = slayout
        self._s_fractal_size = s_fractal_size
        self._pad = pad
        # Build MLIR attrs eagerly only if a context is already active.
        if _has_context():
            self._build_attrs()

    def _build_attrs(self):
        self._bl = _pto.BLayoutAttr.get(getattr(_pto.BLayout, self._blayout))
        self._sl = _pto.SLayoutAttr.get(getattr(_pto.SLayout, self._slayout))
        self._pd = _pto.PadValueAttr.get(getattr(_pto.PadValue, self._pad))

    @property
    def attr(self):
        if not hasattr(self, "_bl"):
            self._build_attrs()
        return _pto.TileBufConfigAttr.get(
            self._bl, self._sl, self._s_fractal_size, self._pd
        )


def _default_tile_config(memory_space, shape):
    space = memory_space.upper()
    # Defaults mirror the explicit configs used by the verbose matmul builder.
    if space == "MAT":
        if len(shape) >= 1 and shape[0] == 1:
            return TileBufConfig(
                blayout="RowMajor",
                slayout="NoneBox",
                s_fractal_size=_pto.TileConfig.fractalABSize,
            )
        return TileBufConfig(
            blayout="ColMajor",
            slayout="RowMajor",
            s_fractal_size=_pto.TileConfig.fractalABSize,
        )
    if space == "LEFT":
        return TileBufConfig(
            blayout="RowMajor",
            slayout="RowMajor",
            s_fractal_size=_pto.TileConfig.fractalABSize,
        )
    if space == "RIGHT":
        return TileBufConfig(
            blayout="RowMajor",
            slayout="ColMajor",
            s_fractal_size=_pto.TileConfig.fractalABSize,
        )
    if space == "ACC":
        return TileBufConfig(
            blayout="ColMajor",
            slayout="RowMajor",
            s_fractal_size=_pto.TileConfig.fractalCSize,
        )
    if space == "BIAS":
        return TileBufConfig(
            blayout="RowMajor",
            slayout="NoneBox",
            s_fractal_size=_pto.TileConfig.fractalABSize,
        )
    if space == "VEC":
        return TileBufConfig()
    raise ValueError(
        f"Unsupported memory_space '{memory_space}' for default tile config."
    )


def _make_tile_buf_type(shape, dtype, memory_space, valid_shape, config):
    dtype = _materialize(dtype)
    space = _pto.AddressSpaceAttr.get(getattr(_pto.AddressSpace, memory_space))
    if valid_shape is None:
        valid_shape = shape
    if config is None:
        config = _default_tile_config(memory_space, shape)
    cfg = config.attr if isinstance(config, TileBufConfig) else config
    return _pto.TileBufType.get(shape, dtype, space, valid_shape, cfg)


def TileBufType(*, shape, dtype, memory_space, valid_shape=None, config=None):
    if _has_context():
        return _make_tile_buf_type(shape, dtype, memory_space, valid_shape, config)
    return _LazyType(
        _make_tile_buf_type, (shape, dtype, memory_space, valid_shape, config), {}
    )


__all__ = [
    "_LazyType",
    "_materialize",
    "PtrType",
    "TensorType",
    "SubTensorType",
    "TileBufConfig",
    "TileBufType",
    "bool",
    "float16",
    "float32",
    "int16",
    "int32",
    "ffts_type",
    "uint32",
    "int8",
    "uint8",
]
