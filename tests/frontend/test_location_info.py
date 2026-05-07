from ptodsl import pto, to_ir_module
from ptodsl import scalar as s


dtype = pto.float32
index_dtype = pto.int32
ptr_type = pto.PtrType(dtype)
tile_type = pto.TileBufType(
    shape=[32, 32],
    valid_shape=[-1, -1],
    dtype=dtype,
    memory_space="VEC",
)


@to_ir_module
def kernel(
    x_ptr: ptr_type,
    y_ptr: ptr_type,
    batch_i32: index_dtype,
    n_cols_i32: index_dtype,
) -> None:
    c1 = s.const(1)
    add = c1 + c1
    id = pto.get_block_idx()
    batch = s.index_cast(batch_i32)


def test_location_info_in_asm():
    asm = kernel.operation.get_asm(enable_debug_info=True)
    print(asm)
    # Kernel def — line of the @to_ir_module decorated function definition
    assert 'test_location_info.py":16:0)' in asm
    # Const def
    assert 'test_location_info.py":23:9)' in asm
    # Add def
    assert 'test_location_info.py":24:10)' in asm
    # Block idx def
    assert 'test_location_info.py":25:9)' in asm
    # Index cast def
    assert 'test_location_info.py":26:12)' in asm
