from contextlib import contextmanager

from mlir.dialects import pto as _pto
from mlir.ir import FlatSymbolRefAttr, InsertionPoint, Operation, ShapedType

from .scalar import Value, _unwrap
from .type_def import _materialize
from ..utils.codegen import get_user_code_loc, with_loc


@with_loc
def get_block_idx():
    return Value(_pto.GetBlockIdxOp().result)


@with_loc
def get_subblock_idx():
    return Value(_pto.GetSubBlockIdxOp().result)


@with_loc
def get_subblock_num():
    return Value(_pto.GetSubBlockNumOp().result)


@with_loc
def get_block_num():
    return Value(_pto.GetBlockNumOp().result)


def _resolve_layout_attr(layout):
    if layout is None:
        return None
    if isinstance(layout, str):
        return _pto.LayoutAttr.get(getattr(_pto.Layout, layout))
    return layout


def _resolve_address_space_attr(location):
    if isinstance(location, str):
        return _pto.AddressSpaceAttr.get(getattr(_pto.AddressSpace, location.upper()))
    return location


def _resolve_peer_func_attr(peer_func):
    if hasattr(peer_func, "sym_name"):
        peer_func = peer_func.sym_name
    if isinstance(peer_func, str):
        return FlatSymbolRefAttr.get(peer_func.removeprefix("@"))
    return peer_func


def call(callee, *args):
    return Operation.create(
        "func.call",
        operands=[_unwrap(arg) for arg in args],
        attributes={"callee": _resolve_peer_func_attr(callee)},
    )


@with_loc
def set_ffts(ffts):
    return _pto.SetFFTsOp(_unwrap(ffts))


@with_loc
def sync_set(pipe, event_id, ffts_mode=2):
    return _pto.sync_set(pipe, event_id, ffts_mode)


@with_loc
def sync_wait(pipe, event_id):
    return _pto.sync_wait(pipe, event_id)


@with_loc
def add_ptr(ptr, offset):
    """Return ptr advanced by offset elements, preserving the !pto.ptr type.

    The offset is in elements of the pointer's element type, not bytes.
    """
    return _pto.AddPtrOp(_unwrap(ptr), _unwrap(offset)).result


@with_loc
def as_tensor(*, ptr, shape, strides, layout=None):
    """Create a tensor view over a pointer.

    The result type is inferred: dtype comes from the pointer's element type,
    rank from ``len(shape)``.  All dimensions are dynamic (``?``).
    """
    ptr_val = _unwrap(ptr)
    shape_vals = [_unwrap(v) for v in shape]
    stride_vals = [_unwrap(v) for v in strides]
    elem_type = _pto.PtrType(ptr_val.type).element_type
    tensor_type = _pto.TensorViewType.get(len(shape_vals), elem_type)
    kwargs = {}
    layout_attr = _resolve_layout_attr(layout)
    if layout_attr is not None:
        kwargs["layout"] = layout_attr
    return _pto.MakeTensorViewOp(
        tensor_type, ptr_val, shape_vals, stride_vals, **kwargs
    ).result


@with_loc
def slice_view(*, source, offsets, sizes):
    """Create a partition (tile) view into *source*.

    The result type is fully inferred: dtype from *source*'s element type,
    rank from ``len(sizes)``, all dims dynamic (``?``).  ptoas infers the
    concrete shape from the paired ``alloc_tile`` / ``tile_buf``.
    """
    src_val = _unwrap(source)
    offset_vals = [_unwrap(v) for v in offsets]
    size_vals = [_unwrap(v) for v in sizes]
    src_type = src_val.type
    if _pto.TensorViewType.isinstance(src_type):
        elem_type = _pto.TensorViewType(src_type).element_type
    else:
        elem_type = _pto.PartitionTensorViewType(src_type).element_type
    dyn = ShapedType.get_dynamic_size()
    rank = len(sizes)
    subtensor_type = _pto.PartitionTensorViewType.get([dyn] * rank, elem_type)
    return _pto.PartitionViewOp(
        subtensor_type, src_val, offsets=offset_vals, sizes=size_vals
    ).result


@contextmanager
def vector_section():
    with get_user_code_loc():
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


@with_loc
def alloc_tile(tile_type, *, addr=None, valid_row=None, valid_col=None):
    tile_type = _materialize(tile_type)
    kwargs = {}
    if addr is not None:
        kwargs["addr"] = _unwrap(addr)
    if valid_row is not None:
        kwargs["valid_row"] = _unwrap(valid_row)
    if valid_col is not None:
        kwargs["valid_col"] = _unwrap(valid_col)
    return _pto.AllocTileOp(tile_type, **kwargs).result


@with_loc
def declare_tile(tile_type):
    tile_type = _materialize(tile_type)
    return Operation.create("pto.declare_tile", results=[tile_type]).result


def declare_global(tensor_type):
    return _pto.DeclareGlobalOp(tensor_type).result


# %c2v_local = pto.reserve_buffer {
#     name = "c2v_fifo",
#     size = 4096,
#     location = #pto.address_space<vec>,
#     auto = true
# } -> i32
@with_loc
def reserve_buffer(*, name, size, location, auto_alloc=True, base=None):
    """
    - At most one `pto.reserve_buffer` is expected in one function
    - `location` must be a supported local address space
    - Op-level verification requires:
    - `auto = false` must provide `base`
    - `auto = true` must not provide `base`
    """
    # All params are compile time attributes
    # wrap reserve_buffer(name, size, location, auto_alloc, *, base=None, loc=None, ip=None) -> mlir._mlir_libs._mlir.ir.Value
    return _pto.ReserveBufferOp(
        name, size, _resolve_address_space_attr(location), auto_alloc, base=base
    ).result


# %c2v_import = pto.import_reserved_buffer {
#     name = "c2v_fifo",
#     peer_func = @vector_kernel
# } -> i32
@with_loc
def import_reserved_buffer(*, name, peer_func):
    # wrap import_reserved_buffer(name, peer_func, *, loc=None, ip=None) -> mlir._mlir_libs._mlir.ir.Value
    return _pto.ImportReservedBufferOp(name, _resolve_peer_func_attr(peer_func)).result


@with_loc
def aic_initialize_pipe(
    *,
    dir_mask,
    slot_size,
    gm_slot_buffer=None,  # only needed on a2/a3?
    c2v_consumer_buf,
    v2c_consumer_buf,
    id=None,
    nosplit=None,
):
    return _pto.AicInitializePipeOp(
        dir_mask,
        slot_size,
        c2v_consumer_buf=_unwrap(c2v_consumer_buf),
        v2c_consumer_buf=_unwrap(v2c_consumer_buf),
        gm_slot_buffer=_unwrap(gm_slot_buffer),
        id=id,
        nosplit=nosplit,
    )


# pto.aiv_initialize_pipe {dir_mask = 1, slot_size = 1024} (
#    gm_slot_buffer = %gm_slot_buffer : !pto.ptr<f32>,
#    c2v_consumer_buf = %c2v_local : i32,
#    v2c_consumer_buf = %c0_i32 : i32
# )
@with_loc
def aiv_initialize_pipe(
    *,
    dir_mask,
    slot_size,
    gm_slot_buffer=None,  # only needed on a2/a3
    c2v_consumer_buf,
    v2c_consumer_buf,
    id=None,
    nosplit=None,
):
    return _pto.AivInitializePipeOp(
        dir_mask,
        slot_size,
        c2v_consumer_buf=_unwrap(c2v_consumer_buf),
        v2c_consumer_buf=_unwrap(v2c_consumer_buf),
        gm_slot_buffer=_unwrap(gm_slot_buffer),
        id=id,
        nosplit=nosplit,
    )


def initialize_l2g2l_pipe(
    *,
    dir_mask,
    slot_size,
    slot_num,
    gm_addr,
    local_addr=None,
    peer_local_addr=None,
    local_slot_num=None,
    flag_base=None,
    nosplit=None,
):
    """Initialize a local-to-global-to-local pipe handle.

    Returns a `!pto.pipe` value usable with downstream pipe ops. Unlike the
    legacy `aic/aiv_initialize_pipe` (which is hard-capped at one pipe per
    function with DIR_BOTH 4+4 slots on a3), this op accepts an explicit
    ``slot_num`` (e.g. 8) and is the basis for L2 prefetch / GM_FIFO patterns
    used by the reference flash attention performance kernel.
    """
    return _pto.InitializeL2G2LPipeOp(
        dir_mask,
        slot_size,
        slot_num,
        _unwrap(gm_addr),
        local_slot_num=local_slot_num,
        flag_base=flag_base,
        nosplit=nosplit,
        local_addr=_unwrap(local_addr) if local_addr is not None else None,
        peer_local_addr=(
            _unwrap(peer_local_addr) if peer_local_addr is not None else None
        ),
    ).result


# -----------------------------------------------------------------------------
# Pipe-handle generic tpush/tpop/tfree (used with `initialize_l2g2l_pipe`handles, as opposed to the legacy *_to_aic / *_to_aiv
# / *_from_aic / *_from_aiv variants tied to the function-scoped legacy pipe).
# -----------------------------------------------------------------------------
@with_loc
def tpush(tile, pipe_handle, split):
    """Push a tile onto a pipe handle (l2g2l or l2l)."""
    return _pto.TPushOp(_unwrap(tile), _unwrap(pipe_handle), split)


@with_loc
def talloc(entry, pipe_handle, split):
    """Allocate the next global tensor slot on a pipe handle."""
    return _pto.TAllocOp(_unwrap(entry), _unwrap(pipe_handle), split)


@with_loc
def tpop_into(entry, pipe_handle, split):
    """Pop the next pipe entry into an existing tile handle."""
    return _pto.TPopOp(_unwrap(entry), _unwrap(pipe_handle), split)


@with_loc
def tpop(tile_type, pipe_handle, split, *, addr=None):
    """Pop the next tile from a pipe handle.

    The underlying ``pto.tpop`` op is destination-passing: it writes into a
    pre-allocated tile. This wrapper allocates a fresh tile of ``tile_type``,
    pops into it, and returns the tile value.
    """
    kwargs = {}
    if addr is not None:
        kwargs["addr"] = _unwrap(addr)
    dest = _pto.AllocTileOp(tile_type, **kwargs).result
    _pto.TPopOp(dest, _unwrap(pipe_handle), split)
    return dest


@with_loc
def tfree(pipe_handle, split, *, entry=None):
    """Release the slot most recently popped from a pipe handle."""
    kwargs = {}
    if entry is not None:
        kwargs["entry"] = _unwrap(entry)
    return _pto.TFreeOp(_unwrap(pipe_handle), split, **kwargs)


# pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, ..., pad=0>) {split = 0}
@with_loc
def tpush_to_aiv(tile, split, *, id=None):
    return _pto.TPushToAivOp(_unwrap(tile), split, id=id)


@with_loc
def tpush_to_aic(tile, split, *, id=None):
    return _pto.TPushToAicOp(_unwrap(tile), split, id=id)


# %recv_tile = pto.tpop_from_aic {split = 0} -> !pto.tile_buf<loc=vec, ... fractal=512, pad=0>
@with_loc
def tpop_from_aic(tile_type, split, *, id=None):
    return _pto.TPopFromAicOp(tile_type, split, id=id).result


@with_loc
def tpop_from_aiv(tile_type, split, *, id=None):
    return _pto.TPopFromAivOp(tile_type, split, id=id).result


# pto.tfree_from_aic {split = 0}
@with_loc
def tfree_from_aic(split, *, id=None):
    return _pto.TFreeFromAicOp(split, id=id)


@with_loc
def tfree_from_aiv(split, *, id=None):
    return _pto.TFreeFromAivOp(split, id=id)


@with_loc
def load_scalar(result_type, ptr, offset):
    """Load a single scalar element from global memory at ptr[offset]."""
    return _pto.load_scalar(result_type, _unwrap(ptr), _unwrap(offset))


@with_loc
def load(source, dest):
    return _pto.TLoadOp(None, source, dest)


@with_loc
def store(source, dest):
    return _pto.TStoreOp(None, source, dest)


@with_loc
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
    "call",
    "set_ffts",
    "sync_set",
    "sync_wait",
    "add_ptr",
    "as_tensor",
    "slice_view",
    "vector_section",
    "cube_section",
    "alloc_tile",
    "declare_global",
    "declare_tile",
    "reserve_buffer",
    "import_reserved_buffer",
    "aic_initialize_pipe",
    "aiv_initialize_pipe",
    "initialize_l2g2l_pipe",
    "load_scalar",
    "load",
    "store",
    "tpush_to_aiv",
    "tpush_to_aic",
    "tpop_from_aic",
    "tpop_from_aiv",
    "tfree_from_aic",
    "tfree_from_aiv",
    "tpush",
    "talloc",
    "tpop_into",
    "tpop",
    "tfree",
    "print",
]
