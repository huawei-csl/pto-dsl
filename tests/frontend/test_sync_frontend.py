from mlir.ir import Context, Location
from mlir.dialects import pto as mlir_pto

from ptodsl import pto, to_ir_module


def test_barrier_sync_emits_pto_barrier_sync_op():
    @to_ir_module(meta_data=lambda: {})
    def sync_kernel() -> None:
        pto.barrier_sync("vec")
        pto.barrier_sync(mlir_pto.SyncOpTypeAttr.get(mlir_pto.SyncOpType.TMATMUL))

    text = str(sync_kernel)

    assert "pto.barrier_sync[<TVEC>]" in text
    assert "pto.barrier_sync[<TMATMUL>]" in text


def test_barrier_sync_helper_accepts_string_and_attr():
    with Context() as ctx, Location.unknown():
        mlir_pto.register_dialect(ctx, load=True)

        attr = mlir_pto.SyncOpTypeAttr.get(mlir_pto.SyncOpType.TVEC)
        assert str(attr) == "#pto.sync_op_type<TVEC>"
