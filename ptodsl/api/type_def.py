from mlir.dialects import pto as _pto

from . import scalar


def __getattr__(name):
    if name in {
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
    }:
        return getattr(scalar, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def PtrType(dtype):
    return _pto.PtrType.get(dtype)


def TensorType(*, rank, dtype):
    return _pto.TensorViewType.get(rank, dtype)


def SubTensorType(*, shape, dtype):
    return _pto.PartitionTensorViewType.get(shape, dtype)


class TileConfig:
    def __init__(self, blayout="RowMajor", slayout="NoneBox", s_fractal_size=512, pad="Null"):
        # TODO: expose and validate a broader set of tile buffer knobs if PTO adds
        # more layout/padding/fractal settings that should be configurable here.
        self._bl = _pto.BLayoutAttr.get(getattr(_pto.BLayout, blayout))
        self._sl = _pto.SLayoutAttr.get(getattr(_pto.SLayout, slayout))
        self._pd = _pto.PadValueAttr.get(getattr(_pto.PadValue, pad))
        self._s_fractal_size = s_fractal_size

    @property
    def attr(self):
        return _pto.TileBufConfigAttr.get(self._bl, self._sl, self._s_fractal_size, self._pd)


def _default_tile_config(memory_space, shape):
    space = memory_space.upper()
    # Defaults mirror the explicit configs used by the verbose matmul builder.
    if space == "MAT":
        if len(shape) >= 1 and shape[0] == 1:
            return TileConfig(
                blayout="RowMajor",
                slayout="NoneBox",
                s_fractal_size=_pto.TileConfig.fractalABSize,
            )
        return TileConfig(
            blayout="ColMajor",
            slayout="RowMajor",
            s_fractal_size=_pto.TileConfig.fractalABSize,
        )
    if space == "LEFT":
        return TileConfig(
            blayout="RowMajor",
            slayout="RowMajor",
            s_fractal_size=_pto.TileConfig.fractalABSize,
        )
    if space == "RIGHT":
        return TileConfig(
            blayout="RowMajor",
            slayout="ColMajor",
            s_fractal_size=_pto.TileConfig.fractalABSize,
        )
    if space == "ACC":
        return TileConfig(
            blayout="ColMajor",
            slayout="RowMajor",
            s_fractal_size=_pto.TileConfig.fractalCSize,
        )
    if space == "BIAS":
        return TileConfig(
            blayout="RowMajor",
            slayout="NoneBox",
            s_fractal_size=_pto.TileConfig.fractalABSize,
        )
    if space == "VEC":
        return TileConfig()
    raise ValueError(f"Unsupported memory_space '{memory_space}' for default tile config.")


def TileType(*, shape, dtype, memory_space, valid_shape=None, config=None):
    space = _pto.AddressSpaceAttr.get(getattr(_pto.AddressSpace, memory_space))
    if valid_shape is None:
        valid_shape = shape
    if config is None:
        config = _default_tile_config(memory_space, shape)
    cfg = config.attr if isinstance(config, TileConfig) else config
    raw_tile = _pto.TileType.get(shape, dtype)
    if hasattr(raw_tile, "to_buffer"):
        return raw_tile.to_buffer(space, valid_shape=valid_shape, config=cfg)
    # PTOAS still carries TileBufType internally in older builds. Keep a
    # compatibility fallback until all environments rebuild with TileType.to_buffer.
    return _pto.TileBufType.get(shape, dtype, space, valid_shape, cfg)


__all__ = [
    "PtrType",
    "TensorType",
    "SubTensorType",
    "TileConfig",
    "TileType",
    "bool",
    "int8",
    "uint8",
    "float16",
    "bfloat16",
    "float32",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float8_e4m3fn",
    "float8_e5m2",
]
