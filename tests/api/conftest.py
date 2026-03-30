import pytest
from mlir.dialects import pto as mlir_pto
from mlir.ir import Context, Location


@pytest.fixture
def mlir_ctx():
    with Context() as ctx, Location.unknown():
        mlir_pto.register_dialect(ctx, load=True)
        yield ctx
