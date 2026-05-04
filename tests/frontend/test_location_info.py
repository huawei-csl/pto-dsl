from ptodsl import pto, to_ir_module
from ptodsl import scalar as s

def meta_data():
    dtype = pto.float32
    index_dtype = pto.int32
    ptr_type = pto.PtrType(dtype)
    tensor_type = pto.TensorType(rank=2, dtype=dtype)
    subtensor_type = pto.SubTensorType(shape=[32, 32], dtype=dtype)
    tile_cfg = pto.TileBufConfig()
    tile_type = pto.TileBufType(
        shape=[32, 32],
        valid_shape=[-1, -1],
        dtype=dtype,
        memory_space="VEC",
        config=tile_cfg,
    )
    return {
        "ptr_type": ptr_type,
        "index_dtype": index_dtype,
        "tensor_type": tensor_type,
        "subtensor_type": subtensor_type,
        "tile_type": tile_type,
    }

def kernel(
    x_ptr: "ptr_type",
    y_ptr: "ptr_type",
    batch_i32: "index_dtype",
    n_cols_i32: "index_dtype",
) -> None:
    c0 = s.const(0)
    c1 = s.const(1)

    batch = s.index_cast(batch_i32)
    n_cols = s.index_cast(n_cols_i32)

def test_location_info_in_asm():
    asm = to_ir_module(meta_data=meta_data)(kernel).operation.get_asm(
        enable_debug_info=True
    )
    print(asm)
    # Kernel def
    assert 'tests/frontend/test_location_info.py":26:0)' in asm
    # Const def
    assert 'tests/frontend/test_location_info.py":32:9)' in asm

