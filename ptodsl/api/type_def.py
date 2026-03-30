from mlir.dialects import pto as _pto

from .._constexpr import require_static_int, require_static_int_sequence
from . import scalar


def __getattr__(name):
    # MLIR type factories require an active context, so keep dtype aliases lazy
    # and resolve them only when user code accesses them inside PTO/MLIR setup.
    if name in {"bool", "float16", "float32", "int16", "int32", "uint32"}:
        return getattr(scalar, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _resolve_memory_space(memory_space):
    if memory_space is None:
        return None
    if isinstance(memory_space, str):
        return getattr(_pto.AddressSpace, memory_space)
    if hasattr(memory_space, "value") and not isinstance(memory_space, int):
        return int(memory_space.value)
    return memory_space


def PtrType(dtype, memory_space=None):
    resolved = scalar.resolve_type(dtype)
    memory_space = _resolve_memory_space(memory_space)
    if memory_space is None:
        return _pto.PtrType.get(resolved)
    return _pto.PtrType.get(resolved, memory_space)


def VRegType(lanes, dtype):
    return _pto.VRegType.get(require_static_int(lanes, context="VRegType.lanes"), scalar.resolve_type(dtype))


def MaskType():
    return _pto.MaskType.get()


def AlignType():
    return _pto.AlignType.get()


def TensorType(*, rank, dtype):
    return _pto.TensorViewType.get(
        require_static_int(rank, context="TensorType.rank"),
        scalar.resolve_type(dtype),
    )


def SubTensorType(*, shape, dtype):
    return _pto.PartitionTensorViewType.get(
        require_static_int_sequence(shape, context="SubTensorType.shape"),
        scalar.resolve_type(dtype),
    )


class TileBufConfig:
    def __init__(
        self, blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null"
    ):
        # TODO: expose and validate a broader set of tile buffer knobs if PTO adds
        # more layout/padding/fractal settings that should be configurable here.
        self._bl = _pto.BLayoutAttr.get(getattr(_pto.BLayout, blayout))
        self._sl = _pto.SLayoutAttr.get(getattr(_pto.SLayout, slayout))
        self._pd = _pto.PadValueAttr.get(getattr(_pto.PadValue, pad))
        self._s_fractal_size = require_static_int(
            s_fractal_size, context="TileBufConfig.s_fractal_size"
        )

    @property
    def attr(self):
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


def TileBufType(*, shape, dtype, memory_space, valid_shape=None, config=None):
    shape = require_static_int_sequence(shape, context="TileBufType.shape")
    space = _pto.AddressSpaceAttr.get(getattr(_pto.AddressSpace, memory_space))
    if valid_shape is None:
        valid_shape = shape
    else:
        valid_shape = require_static_int_sequence(
            valid_shape, context="TileBufType.valid_shape"
        )
    if config is None:
        config = _default_tile_config(memory_space, shape)
    cfg = config.attr if isinstance(config, TileBufConfig) else config
    return _pto.TileBufType.get(
        shape, scalar.resolve_type(dtype), space, valid_shape, cfg
    )


__all__ = [
    "PtrType",
    "VRegType",
    "MaskType",
    "AlignType",
    "TensorType",
    "SubTensorType",
    "TileBufConfig",
    "TileBufType",
    "bool",
    "float16",
    "float32",
    "int16",
    "int32",
    "uint32",
]
