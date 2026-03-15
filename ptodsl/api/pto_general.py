from contextlib import contextmanager

from mlir.dialects import pto as _pto
from mlir.ir import InsertionPoint

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


class _SectionNamespace:
    def vector(self):
        return vector_section()

    def cube(self):
        return cube_section()


section = _SectionNamespace()


def alloc_tile(tile_type, *, addr=None, valid_row=None, valid_col=None):
    kwargs = {}
    if addr is not None:
        kwargs["addr"] = _unwrap(addr)
    if valid_row is not None:
        kwargs["valid_row"] = _unwrap(valid_row)
    if valid_col is not None:
        kwargs["valid_col"] = _unwrap(valid_col)
    return _pto.AllocTileOp(tile_type, **kwargs).result


def load(source, dest):
    _pto.TLoadOp(None, source, dest)


def store(source, dest):
    _pto.TStoreOp(None, source, dest)


def load_scalar(result_type, ptr, offset):
    return _pto.load_scalar(result_type, _unwrap(ptr), _unwrap(offset))


def store_scalar(ptr, offset, value):
    _pto.store_scalar(_unwrap(ptr), _unwrap(offset), _unwrap(value))


def store_fp(source, fp, dest):
    # This is the PTOAS/PTO-ISA TSTORE_FP path, which currently lowers to the
    # accumulator-side quantized store contract. It is not a generic vector-tile
    # quantize-store helper for mixed-dtype vec -> GM stores.
    _pto.TStoreFPOp(source, fp, dest)


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
    "get_block_idx",
    "get_subblock_idx",
    "get_subblock_num",
    "get_block_num",
    "as_tensor",
    "slice_view",
    "section",
    "vector_section",
    "cube_section",
    "alloc_tile",
    "load",
    "store",
    "load_scalar",
    "store_scalar",
    "store_fp",
    "print",
]
