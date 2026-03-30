import pathlib
from types import SimpleNamespace

import pytest

from ptodsl import Constexpr, const_expr, pto, range_constexpr, to_ir_module
from ptodsl import scalar as s
from ptodsl.compiler.jit import JitWrapper


const = s.const


class _FakeType:
    def __init__(self, text):
        self._text = text

    def __str__(self):
        return self._text


def test_to_ir_module_returns_specializer_and_prunes_constexpr_signature():
    seen = {}

    def meta_data(TILE, UNROLL=2):
        seen["tile"] = TILE
        seen["unroll"] = UNROLL
        dtype = pto.float32
        return {
            "index_dtype": pto.int32,
            "tile_type": pto.TileBufType(
                shape=[1, TILE // 2],
                valid_shape=[1, TILE // 2],
                dtype=dtype,
                memory_space="VEC",
            ),
        }

    @to_ir_module(meta_data=meta_data)
    def templated_kernel(
        n: "index_dtype",
        TILE: Constexpr[int],
        UNROLL: Constexpr[int] = 2,
    ) -> None:
        with pto.vector_section():
            if const_expr(TILE % 128 == 0):
                for _ in range_constexpr(UNROLL):
                    pto.alloc_tile(tile_type)
            else:
                pto.alloc_tile(tile_type)

    assert callable(templated_kernel)

    module = templated_kernel(TILE=128, UNROLL=3)
    text = str(module)

    assert seen == {"tile": 128, "unroll": 3}
    assert "func.func @templated_kernel(%arg0: i32)" in text
    assert "scf.if" not in text
    assert "scf.for" not in text
    assert text.count("pto.alloc_tile") == 3


def test_to_ir_module_rejects_missing_constexpr_arguments():
    def meta_data(TILE):
        return {"index_dtype": pto.int32}

    @to_ir_module(meta_data=meta_data)
    def templated_kernel(n: "index_dtype", TILE: Constexpr[int]) -> None:
        return None

    with pytest.raises(TypeError, match="Missing required constexpr arguments: TILE"):
        templated_kernel()


def test_constexpr_helpers_reject_dynamic_values():
    dynamic = s.Value(object())

    with pytest.raises(TypeError, match="const_expr"):
        const_expr(dynamic)

    with pytest.raises(TypeError, match="range_constexpr"):
        list(range_constexpr(dynamic))


def test_type_builders_reject_dynamic_static_dimensions():
    with pytest.raises(TypeError, match="TensorType.rank"):
        pto.TensorType(rank=s.Value(object()), dtype=pto.float32)

    with pytest.raises(TypeError, match="TileBufType.shape"):
        pto.TileBufType(
            shape=[1, s.Value(object())],
            valid_shape=[1, 1],
            dtype=pto.float32,
            memory_space="VEC",
        )


def test_jit_prunes_constexpr_parameters_from_generated_caller_cpp():
    def mixed_kernel(
        data: "ptr_i8",
        count: "i64_type",
        TILE: Constexpr[int],
    ) -> None:
        return None

    wrapper = JitWrapper(mixed_kernel, meta_data=lambda TILE: {}, block_dim=7)
    wrapper._arg_types = [
        _FakeType("!pto.ptr<i8>"),
        _FakeType("i64"),
    ]

    caller_cpp = wrapper._generate_caller_cpp("generated.cpp")

    assert "TILE" not in caller_cpp
    assert (
        'extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *data, '
        "int64_t count)"
    ) in caller_cpp
    assert "mixed_kernel<<<blockDim, nullptr, stream>>>((int8_t *)data, count);" in caller_cpp


def test_jit_reuses_specialization_for_identical_constexpr_bindings(monkeypatch):
    class _FakeLib:
        def __init__(self):
            self.calls = []

        def call_kernel(self, *args):
            self.calls.append(args)

    def templated_kernel(n: "index_dtype", TILE: Constexpr[int]) -> None:
        return None

    wrapper = JitWrapper(templated_kernel, meta_data=lambda TILE: {}, block_dim=4)
    build_calls = []
    built_libs = []

    def fake_build(constexpr_bindings):
        build_calls.append(dict(constexpr_bindings))
        lib = _FakeLib()
        built_libs.append(lib)
        return SimpleNamespace(
            arg_types=[_FakeType("i32")],
            lib=lib,
            lib_path=pathlib.Path("/tmp/fake.so"),
            output_dir=pathlib.Path("/tmp"),
        )

    monkeypatch.setattr(wrapper, "_build", fake_build)

    wrapper(7, TILE=32, stream_ptr=0)
    wrapper(9, TILE=32, stream_ptr=0)
    wrapper(11, TILE=64, stream_ptr=0)

    assert build_calls == [{"TILE": 32}, {"TILE": 64}]
    assert len(built_libs[0].calls) == 2
    assert len(built_libs[1].calls) == 1
