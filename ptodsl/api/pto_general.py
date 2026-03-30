from contextlib import contextmanager

from mlir.dialects import pto as _pto
from mlir.ir import InsertionPoint

from .._constexpr import require_static_int_sequence
from . import scalar, type_def
from .scalar import Value, _unwrap


def get_block_idx():
    return Value(_pto.GetBlockIdxOp().result)


def get_subblock_idx():
    return Value(_pto.GetSubBlockIdxOp().result)


def get_subblock_num():
    return Value(_pto.GetSubBlockNumOp().result)


def get_block_num():
    return Value(_pto.GetBlockNumOp().result)


def _resolve_layout_attr(layout):
    if layout is None:
        return None
    if isinstance(layout, str):
        return _pto.LayoutAttr.get(getattr(_pto.Layout, layout))
    return layout


def _is_static_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _shape_factor(value):
    if _is_static_int(value):
        return value
    if isinstance(value, Value):
        return value
    if hasattr(value, "raw"):
        return value.raw
    return value


def _mul_shape_values(lhs, rhs):
    lhs = _shape_factor(lhs)
    rhs = _shape_factor(rhs)
    if _is_static_int(lhs) and _is_static_int(rhs):
        return lhs * rhs
    if _is_static_int(lhs) and lhs == 1:
        return rhs
    if _is_static_int(rhs) and rhs == 1:
        return lhs
    lhs_value = scalar.const(lhs) if _is_static_int(lhs) else scalar.wrap_value(lhs)
    rhs_value = scalar.const(rhs) if _is_static_int(rhs) else scalar.wrap_value(rhs)
    return lhs_value * rhs_value


def _as_index_operand(value):
    if _is_static_int(value):
        return scalar.const(value)
    return value


def _normalize_index_operands(values):
    return [_as_index_operand(value) for value in values]


def _infer_compact_row_major_strides(shape):
    shape = list(shape)
    if not shape:
        raise ValueError("`make_tensor` requires a non-empty shape.")
    strides = [1] * len(shape)
    running = 1
    for idx in range(len(shape) - 1, -1, -1):
        strides[idx] = running
        if idx > 0:
            running = _mul_shape_values(running, shape[idx])
    return strides


def _unwrap_tile_type(tile_type):
    if hasattr(tile_type, "raw_type"):
        return tile_type.raw_type
    return tile_type


def _resolve_tensor_element_type(tensor_type, dtype):
    if dtype is not None:
        return scalar.resolve_type(dtype)
    if hasattr(tensor_type, "element_type"):
        return tensor_type.element_type
    raise TypeError(
        "`make_tensor` could not infer the element type from `type=`; pass `dtype=` explicitly."
    )


class TensorView:
    def __init__(self, raw, *, element_type):
        self.raw = _unwrap(raw)
        self._element_type = element_type

    def __getattr__(self, item):
        return getattr(self.raw, item)

    def __repr__(self):
        return str(self.raw)

    def slice(self, offsets, sizes, *, static_shape=None):
        raw_sizes = list(sizes)
        offsets = _normalize_index_operands(offsets)
        sizes = _normalize_index_operands(raw_sizes)
        if static_shape is None:
            if all(_is_static_int(size) for size in raw_sizes):
                inferred_shape = require_static_int_sequence(
                    raw_sizes, context="TensorView.slice.sizes"
                )
            else:
                raise TypeError(
                    "`TensorView.slice(..., sizes=...)` requires `static_shape=` when "
                    "sizes include dynamic PTODSL/MLIR values."
                )
        else:
            inferred_shape = require_static_int_sequence(
                static_shape, context="TensorView.slice.static_shape"
            )
        subtensor_type = type_def.SubTensorType(
            shape=inferred_shape,
            dtype=self._element_type,
        )
        return TensorView(
            slice_view(subtensor_type, source=self.raw, offsets=offsets, sizes=sizes),
            element_type=self._element_type,
        )


class TileBufferSpec:
    def __init__(self, raw_type):
        self.raw_type = raw_type

    def __repr__(self):
        return str(self.raw_type)

    def alloc(self, *, addr=None, valid_row=None, valid_col=None):
        return TileBuffer(
            alloc_tile(
                self.raw_type,
                addr=addr,
                valid_row=valid_row,
                valid_col=valid_col,
            )
        )


class TileBuffer:
    def __init__(self, raw):
        self.raw = _unwrap(raw)

    def __getattr__(self, item):
        return getattr(self.raw, item)

    def __repr__(self):
        return str(self.raw)

    def load_from(self, view):
        load(view, self)
        return self

    def store_to(self, view):
        store(self, view)
        return self


def ptr(dtype, space=None):
    return type_def.PtrType(dtype, memory_space=space)


def make_tensor(ptr, *, shape, strides=None, dtype=None, type=None, layout=None):
    if type is None:
        if dtype is None:
            raise TypeError("`make_tensor` requires `dtype=` when `type=` is omitted.")
        type = type_def.TensorType(rank=len(shape), dtype=dtype)
    element_type = _resolve_tensor_element_type(type, dtype)
    shape = _normalize_index_operands(shape)
    if strides is None:
        strides = _infer_compact_row_major_strides(shape)
    strides = _normalize_index_operands(strides)
    return TensorView(
        as_tensor(type, ptr=ptr, shape=shape, strides=strides, layout=layout),
        element_type=element_type,
    )


def make_tile_buffer(dtype, shape, *, space, valid_shape=None, config=None):
    return TileBufferSpec(
        type_def.TileBufType(
            shape=shape,
            dtype=dtype,
            memory_space=space,
            valid_shape=valid_shape,
            config=config,
        )
    )


def as_tensor(tensor_type, *, ptr, shape, strides, layout=None):
    shape_vals = [_unwrap(v) for v in shape]
    stride_vals = [_unwrap(v) for v in strides]
    kwargs = {}
    layout_attr = _resolve_layout_attr(layout)
    if layout_attr is not None:
        kwargs["layout"] = layout_attr
    return _pto.MakeTensorViewOp(
        tensor_type, _unwrap(ptr), shape_vals, stride_vals, **kwargs
    ).result


def slice_view(subtensor_type, *, source, offsets, sizes):
    offset_vals = [_unwrap(v) for v in offsets]
    size_vals = [_unwrap(v) for v in sizes]
    return _pto.PartitionViewOp(
        subtensor_type, _unwrap(source), offsets=offset_vals, sizes=size_vals
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


def alloc_tile(tile_type, *, addr=None, valid_row=None, valid_col=None):
    kwargs = {}
    if addr is not None:
        kwargs["addr"] = _unwrap(addr)
    if valid_row is not None:
        kwargs["valid_row"] = _unwrap(valid_row)
    if valid_col is not None:
        kwargs["valid_col"] = _unwrap(valid_col)
    return _pto.AllocTileOp(_unwrap_tile_type(tile_type), **kwargs).result


def load(source, dest):
    _pto.TLoadOp(None, _unwrap(source), _unwrap(dest))


def store(source, dest):
    _pto.TStoreOp(None, _unwrap(source), _unwrap(dest))


def print(format, scalar):
    """
    Example:
    `print("hello %d\n", const(5))`
    is equivalent to
    `cce::printf("hello%d\n", 5);`

    NOTE: may not print if the print buffer is full from previous
    prints (typical when printing big tiles).
    """
    if isinstance(scalar, Value):
        scalar = _unwrap(scalar)

    _pto.print_(format, scalar)


__all__ = [
    "TensorView",
    "TileBuffer",
    "TileBufferSpec",
    "get_block_idx",
    "get_subblock_idx",
    "get_subblock_num",
    "get_block_num",
    "ptr",
    "make_tensor",
    "make_tile_buffer",
    "as_tensor",
    "slice_view",
    "vector_section",
    "cube_section",
    "alloc_tile",
    "load",
    "store",
    "print",
]
