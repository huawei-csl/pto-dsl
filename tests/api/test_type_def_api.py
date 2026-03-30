import pytest
from mlir.dialects import pto as mlir_pto

from ptodsl import scalar as s
from ptodsl.api import type_def


def test_basic_type_builders_render_expected_types(mlir_ctx):
    assert str(type_def.PtrType(type_def.float32)) == "!pto.ptr<f32, gm>"
    assert str(type_def.PtrType(type_def.float32, memory_space="VEC")) == "!pto.ptr<f32, ub>"
    assert str(type_def.VRegType(64, type_def.float32)) == "!pto.vreg<64xf32>"
    assert str(type_def.MaskType()) == "!pto.mask"
    assert str(type_def.AlignType()) == "!pto.align"
    assert str(type_def.TensorType(rank=2, dtype=type_def.float32)) == "!pto.tensor_view<?x?xf32>"
    assert (
        str(type_def.SubTensorType(shape=[32, 32], dtype=type_def.float32))
        == "!pto.partition_tensor_view<32x32xf32>"
    )


def test_tile_buffer_defaults_match_explicit_configs(mlir_ctx):
    vec_default = type_def.TileBufType(
        shape=[32, 32],
        dtype=type_def.float32,
        memory_space="VEC",
    )
    vec_explicit = type_def.TileBufType(
        shape=[32, 32],
        dtype=type_def.float32,
        memory_space="VEC",
        config=type_def.TileBufConfig(),
    )

    mat_default = type_def.TileBufType(
        shape=[1, 64],
        dtype=type_def.float32,
        memory_space="MAT",
    )
    mat_explicit = type_def.TileBufType(
        shape=[1, 64],
        dtype=type_def.float32,
        memory_space="MAT",
        config=type_def.TileBufConfig(
            blayout="RowMajor",
            slayout="NoneBox",
            s_fractal_size=mlir_pto.TileConfig.fractalABSize,
        ),
    )

    assert str(vec_default) == str(vec_explicit)
    assert str(mat_default) == str(mat_explicit)


def test_static_type_builders_reject_dynamic_values(mlir_ctx):
    dynamic = s.Value(object())

    with pytest.raises(TypeError, match="TensorType.rank"):
        type_def.TensorType(rank=dynamic, dtype=type_def.float32)

    with pytest.raises(TypeError, match="SubTensorType.shape"):
        type_def.SubTensorType(shape=[32, dynamic], dtype=type_def.float32)

    with pytest.raises(TypeError, match="TileBufType.shape"):
        type_def.TileBufType(
            shape=[1, dynamic],
            dtype=type_def.float32,
            memory_space="VEC",
        )

    with pytest.raises(TypeError, match="TileBufType.valid_shape"):
        type_def.TileBufType(
            shape=[1, 64],
            valid_shape=[1, dynamic],
            dtype=type_def.float32,
            memory_space="VEC",
        )
