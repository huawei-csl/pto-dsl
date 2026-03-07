from contextlib import contextmanager
from typing import Sequence

from mlir.dialects import pto as _pto
from mlir.ir import InsertionPoint

from .control_flow import cond, for_range, if_context
from . import scalar
from .scalar import Value, _unwrap, wrap_value


def __getattr__(name):
    # MLIR type factories require an active context, so keep dtype aliases lazy
    # and resolve them only when user code accesses them inside PTO/MLIR setup.
    if name in {"bool", "float16", "float32", "int16", "int32"}:
        return getattr(scalar, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def PtrType(dtype):
    return _pto.PtrType.get(dtype)


def TensorType(*, rank, dtype):
    return _pto.TensorViewType.get(rank, dtype)


def SubTensorType(*, shape, dtype):
    return _pto.PartitionTensorViewType.get(shape, dtype)


class TileBufConfig:
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
    raise ValueError(f"Unsupported memory_space '{memory_space}' for default tile config.")


def TileBufType(*, shape, dtype, memory_space, valid_shape=None, config=None):
    space = _pto.AddressSpaceAttr.get(getattr(_pto.AddressSpace, memory_space))
    if valid_shape is None:
        valid_shape = shape
    if config is None:
        config = _default_tile_config(memory_space, shape)
    cfg = config.attr if isinstance(config, TileBufConfig) else config
    return _pto.TileBufType.get(shape, dtype, space, valid_shape, cfg)


def get_block_idx():
    return Value(_pto.GetBlockIdxOp().result)


def get_subblock_idx():
    return Value(_pto.GetSubBlockIdxOp().result)


def get_subblock_num():
    return Value(_pto.GetSubBlockNumOp().result)


def get_block_num():
    return Value(_pto.GetBlockNumOp().result)


def as_tensor(tensor_type, *, ptr, shape, strides):
    shape_vals = [_unwrap(v) for v in shape]
    stride_vals = [_unwrap(v) for v in strides]
    return _pto.MakeTensorViewOp(tensor_type, _unwrap(ptr), shape_vals, stride_vals).result


def slice_view(subtensor_type, *, source, offsets, sizes):
    offset_vals = [_unwrap(v) for v in offsets]
    size_vals = [_unwrap(v) for v in sizes]
    return _pto.PartitionViewOp(
        subtensor_type, source, offsets=offset_vals, sizes=size_vals
    ).result


@contextmanager
def vector_section():
    section = _pto.SectionVectorOp()
    block = section.body.blocks.append()
    with InsertionPoint(block):
        yield


@contextmanager
def cube_section():
    section = _pto.SectionCubeOp()
    block = section.body.blocks.append()
    with InsertionPoint(block):
        yield


def alloc_tile(tile_type, *, valid_row=None, valid_col=None):
    kwargs = {}
    if valid_row is not None:
        kwargs["valid_row"] = _unwrap(valid_row)
    if valid_col is not None:
        kwargs["valid_col"] = _unwrap(valid_col)
    return _pto.AllocTileOp(tile_type, **kwargs).result


def load(source, dest):
    _pto.TLoadOp(None, source, dest)


def store(source, dest):
    _pto.TStoreOp(None, source, dest)


def _resolve_sync_op(sync_op):
    if isinstance(sync_op, str):
        normalized = sync_op.strip().upper()
        if not normalized.startswith("T"):
            normalized = f"T{normalized}"
        try:
            return getattr(_pto, normalized)
        except AttributeError as exc:
            raise ValueError(f"Unsupported sync op type '{sync_op}'.") from exc
    return sync_op


def _resolve_event_id(event_id):
    if isinstance(event_id, int):
        if event_id < 0 or event_id > 7:
            raise ValueError(f"event_id must be in range [0, 7], got {event_id}.")
        return getattr(_pto, f"EVENT_ID{event_id}")
    return event_id


def record_event(record_op, wait_op, event_id: int | Sequence[int] = 0):
    if not isinstance(event_id, int):
        for eid in event_id:
            _pto.record_event(
                _resolve_sync_op(record_op),
                _resolve_sync_op(wait_op),
                _resolve_event_id(eid),
            )
    else:
        _pto.record_event(
            _resolve_sync_op(record_op),
            _resolve_sync_op(wait_op),
            _resolve_event_id(event_id),
        )


def wait_event(record_op, wait_op, event_id: int | Sequence[int] = 0):
    if not isinstance(event_id, int):
        for eid in event_id:
            _pto.wait_event(
                _resolve_sync_op(record_op),
                _resolve_sync_op(wait_op),
                _resolve_event_id(eid),
            )
    else:
        _pto.wait_event(
            _resolve_sync_op(record_op),
            _resolve_sync_op(wait_op),
            _resolve_event_id(event_id),
        )


def record_wait_pair(record_op, wait_op, event_id: int | Sequence[int] = 0):
    record = _resolve_sync_op(record_op)
    wait = _resolve_sync_op(wait_op)
    event = _resolve_event_id(event_id)
    _pto.record_event(record, wait, event)
    _pto.wait_event(record, wait, event)


def barrier(sync_op):
    _pto.barrier(_resolve_sync_op(sync_op))


__all__ = [
    "Value",
    "wrap_value",
    "bool",
    "float16",
    "float32",
    "int16",
    "int32",
    "PtrType",
    "TensorType",
    "SubTensorType",
    "TileBufConfig",
    "TileBufType",
    "get_block_idx",
    "get_subblock_idx",
    "get_subblock_num",
    "get_block_num",
    "as_tensor",
    "slice_view",
    "vector_section",
    "cube_section",
    "for_range",
    "if_context",
    "cond",
    "alloc_tile",
    "load",
    "store",
    "record_event",
    "wait_event",
    "record_wait_pair",
    "barrier",
]
