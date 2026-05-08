from ptodsl import pto, to_ir_module
from ptodsl import scalar as s


@to_ir_module
def kernel(
    batch_i32: pto.int32,
) -> None:
    c1 = s.const(1)
    add = c1 + c1
    id = pto.get_block_idx()
    batch = s.index_cast(batch_i32)


def test_location_info_in_asm():
    asm = kernel.operation.get_asm(enable_debug_info=True)
    print(asm)
    # Kernel def — line of the @to_ir_module decorated function definition
    assert 'test_location_info.py":5:0)' in asm
    # Const def
    assert 'test_location_info.py":9:9)' in asm
    # Add def
    assert 'test_location_info.py":10:10)' in asm
    # Block idx def
    assert 'test_location_info.py":11:9)' in asm
    # Index cast def
    assert 'test_location_info.py":12:12)' in asm
