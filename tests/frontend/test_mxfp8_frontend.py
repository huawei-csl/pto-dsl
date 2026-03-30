import types

import ptodsl.language as pto


class _StubType:
    @staticmethod
    def get():
        return object()


def test_mxfp8_family_uses_e5m2_data_and_e8m0_scale(monkeypatch):
    stub_ir = types.SimpleNamespace(
        Float8E5M2Type=_StubType,
        Float8E8M0FNUType=_StubType,
        Float8E4M3FNType=_StubType,
    )
    monkeypatch.setattr(pto, "mlir_ir", stub_ir)

    mx = pto.mxfp8

    assert mx.lhs is not None
    assert mx.rhs is not None
    assert mx.data is not None
    assert mx.scale is not None
    assert mx.acc is not None
    assert mx.scale_k(64) == 2


def test_float8_aliases_accept_common_mlir_ctor_names(monkeypatch):
    stub_ir = types.SimpleNamespace(
        Float8E4M3FNType=_StubType,
        Float8E5M2Type=_StubType,
        Float8E8M0FNUType=_StubType,
    )
    monkeypatch.setattr(pto, "mlir_ir", stub_ir)

    assert pto.fp8_e4m3 is not None
    assert pto.fp8_e5m2 is not None
    assert pto.fp8_e8m0 is not None


def test_make_mxfp8_accepts_mixed_lhs_rhs_variants(monkeypatch):
    stub_ir = types.SimpleNamespace(
        Float8E4M3FNType=_StubType,
        Float8E5M2Type=_StubType,
        Float8E8M0FNUType=_StubType,
    )
    monkeypatch.setattr(pto, "mlir_ir", stub_ir)

    mx = pto.make_mxfp8(lhs="e4m3", rhs="e5m2")

    assert mx.lhs is not None
    assert mx.rhs is not None
    assert mx.scale is not None
