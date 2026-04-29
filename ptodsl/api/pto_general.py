from contextlib import contextmanager

from mlir.dialects import pto as _pto
from mlir.ir import FlatSymbolRefAttr, InsertionPoint, Operation

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


def set_ffts(ffts):
    return _pto.SetFFTsOp(_unwrap(ffts))


def add_ptr(ptr, offset):
    """Return ptr advanced by offset elements, preserving the !pto.ptr type.

    The offset is in elements of the pointer's element type, not bytes.
    """
    return _pto.AddPtrOp(_unwrap(ptr), _unwrap(offset)).result


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


def alloc_tile(tile_type, *, addr=None, valid_row=None, valid_col=None):
    kwargs = {}
    if addr is not None:
        kwargs["addr"] = _unwrap(addr)
    if valid_row is not None:
        kwargs["valid_row"] = _unwrap(valid_row)
    if valid_col is not None:
        kwargs["valid_col"] = _unwrap(valid_col)
    return _pto.AllocTileOp(tile_type, **kwargs).result


def declare_tile(tile_type):
    return Operation.create("pto.declare_tile", results=[tile_type]).result


# %c2v_local = pto.reserve_buffer {
#     name = "c2v_fifo",
#     size = 4096,
#     location = #pto.address_space<vec>,
#     auto = true
# } -> i32
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
def import_reserved_buffer(*, name, peer_func):
    # wrap import_reserved_buffer(name, peer_func, *, loc=None, ip=None) -> mlir._mlir_libs._mlir.ir.Value
    return _pto.ImportReservedBufferOp(name, _resolve_peer_func_attr(peer_func)).result


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
    local_addr,
    peer_local_addr=None,
    local_slot_num=None,
    flag_base=None,
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
        _unwrap(local_addr),
        local_slot_num=local_slot_num,
        flag_base=flag_base,
        peer_local_addr=(
            _unwrap(peer_local_addr) if peer_local_addr is not None else None
        ),
    ).result


# -----------------------------------------------------------------------------
# Pipe-handle generic tpush/tpop/tfree (used with `initialize_l2g2l_pipe`handles, as opposed to the legacy *_to_aic / *_to_aiv
# / *_from_aic / *_from_aiv variants tied to the function-scoped legacy pipe).
# -----------------------------------------------------------------------------
def tpush(tile, pipe_handle, split):
    """Push a tile onto a pipe handle (l2g2l or l2l)."""
    return _pto.TPushOp(_unwrap(tile), _unwrap(pipe_handle), split)


def tpop_into(entry, pipe_handle, split):
    """Pop the next pipe entry into an existing tile handle."""
    return _pto.TPopOp(_unwrap(entry), _unwrap(pipe_handle), split)


def tpop(tile_type, pipe_handle, split):
    """Pop the next tile from a pipe handle.

    The underlying ``pto.tpop`` op is destination-passing: it writes into a
    pre-allocated tile. This wrapper allocates a fresh tile of ``tile_type``,
    pops into it, and returns the tile value.
    """
    dest = _pto.AllocTileOp(tile_type).result
    _pto.TPopOp(dest, _unwrap(pipe_handle), split)
    return dest


def tfree(pipe_handle, split):
    """Release the slot most recently popped from a pipe handle."""
    return _pto.TFreeOp(_unwrap(pipe_handle), split)


# pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, ..., pad=0>) {split = 0}
def tpush_to_aiv(tile, split, *, id=None):
    return _pto.TPushToAivOp(_unwrap(tile), split, id=id)


def tpush_to_aic(tile, split, *, id=None):
    return _pto.TPushToAicOp(_unwrap(tile), split, id=id)


# %recv_tile = pto.tpop_from_aic {split = 0} -> !pto.tile_buf<loc=vec, ... fractal=512, pad=0>
def tpop_from_aic(tile_type, split, *, id=None):
    return _pto.TPopFromAicOp(tile_type, split, id=id).result


def tpop_from_aiv(tile_type, split, *, id=None):
    return _pto.TPopFromAivOp(tile_type, split, id=id).result


# pto.tfree_from_aic {split = 0}
def tfree_from_aic(split, *, id=None):
    return _pto.TFreeFromAicOp(split, id=id)


def tfree_from_aiv(split, *, id=None):
    return _pto.TFreeFromAivOp(split, id=id)


def load_scalar(result_type, ptr, offset):
    """Load a single scalar element from global memory at ptr[offset]."""
    return _pto.load_scalar(result_type, _unwrap(ptr), _unwrap(offset))


def load(source, dest):
    _pto.TLoadOp(None, source, dest)


def store(source, dest):
    _pto.TStoreOp(None, source, dest)


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
    "add_ptr",
    "as_tensor",
    "slice_view",
    "vector_section",
    "cube_section",
    "alloc_tile",
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
    "tpop_into",
    "tpop",
    "tfree",
    "print",
]
