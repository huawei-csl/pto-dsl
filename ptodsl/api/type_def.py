from mlir.dialects import pto as _pto

from . import scalar


def __getattr__(name):
    # MLIR type factories require an active context, so keep dtype aliases lazy
    # and resolve them only when user code accesses them inside PTO/MLIR setup.
    if name in {
        "bool",
        "float16",
        "float32",
        "int8",
        "uint8",
        "int16",
        "int32",
        "uint32",
        "int64",
    }:
        return getattr(scalar, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def PtrType(dtype):
    return _pto.PtrType.get(dtype)


def TensorType(*, rank, dtype):
    return _pto.TensorViewType.get(rank, dtype)


def SubTensorType(*, shape, dtype):
    return _pto.PartitionTensorViewType.get(shape, dtype)


class TileBufConfig:
    def __init__(
        self, blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null"
    ):
        # TODO: expose and validate a broader set of tile buffer knobs if PTO adds
        # more layout/padding/fractal settings that should be configurable here.
        self._bl = _pto.BLayoutAttr.get(getattr(_pto.BLayout, blayout))
        self._sl = _pto.SLayoutAttr.get(getattr(_pto.SLayout, slayout))
        self._pd = _pto.PadValueAttr.get(getattr(_pto.PadValue, pad))
        self._s_fractal_size = s_fractal_size

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
    space = _pto.AddressSpaceAttr.get(getattr(_pto.AddressSpace, memory_space))
    if valid_shape is None:
        valid_shape = shape
    if config is None:
        config = _default_tile_config(memory_space, shape)
    cfg = config.attr if isinstance(config, TileBufConfig) else config
    return _pto.TileBufType.get(shape, dtype, space, valid_shape, cfg)


__all__ = [
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
    "uint32",
]
