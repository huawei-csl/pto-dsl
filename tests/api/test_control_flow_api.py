import pytest
from mlir.ir import IndexType

from ptodsl import pto, scalar as s, to_ir_module
from ptodsl.api import control_flow


const = s.const


def test_constexpr_helpers_accept_static_inputs_and_reject_dynamic_values():
    assert control_flow.const_expr(3) is True
    assert list(control_flow.range_constexpr(3)) == [0, 1, 2]
    assert list(control_flow.range_constexpr(1, 5, 2)) == [1, 3]

    dynamic = s.Value(object())
    with pytest.raises(TypeError, match="const_expr"):
        control_flow.const_expr(dynamic)

    with pytest.raises(TypeError, match="range_constexpr"):
        list(control_flow.range_constexpr(dynamic))


def test_range_if_context_and_cond_emit_scf_ops():
    def meta_data():
        return {
            "index_t": IndexType.get(),
            "tile_t": pto.TileBufType(
                shape=[1, 64],
                valid_shape=[1, 64],
                dtype=pto.float32,
                memory_space="VEC",
            ),
        }

    @to_ir_module(meta_data=meta_data)
    def kernel(n: "index_t") -> None:
        c0 = const(0)
        c1 = const(1)
        c2 = const(2)

        with pto.vector_section():
            for i in control_flow.range(c0, c2, c1):
                with control_flow.if_context(s.gt(i, c0)):
                    pto.alloc_tile(tile_t)
            control_flow.cond(
                s.gt(n, c0),
                lambda: pto.alloc_tile(tile_t),
                lambda: pto.alloc_tile(tile_t),
            )

    text = str(kernel)

    assert "scf.for" in text
    assert text.count("scf.if") == 2
    assert text.count("pto.alloc_tile") == 3
