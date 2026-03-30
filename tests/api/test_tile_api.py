from types import SimpleNamespace

from ptodsl.api import tile as tile_api


class _Box:
    def __init__(self, raw):
        self.raw = raw


CALL_CASES = [
    ("mov", lambda: tile_api.mov(_Box("src"), _Box("dst")), "TMovOp", (None, "src", "dst"), {}),
    ("add", lambda: tile_api.add(_Box("lhs"), _Box("rhs"), _Box("out")), "TAddOp", ("lhs", "rhs", "out"), {}),
    ("sub", lambda: tile_api.sub(_Box("lhs"), _Box("rhs"), _Box("out")), "TSubOp", ("lhs", "rhs", "out"), {}),
    ("div", lambda: tile_api.div(_Box("lhs"), _Box("rhs"), _Box("out")), "TDivOp", ("lhs", "rhs", "out"), {}),
    ("mul", lambda: tile_api.mul(_Box("lhs"), _Box("rhs"), _Box("out")), "TMulOp", ("lhs", "rhs", "out"), {}),
    ("or_", lambda: tile_api.or_(_Box("lhs"), _Box("rhs"), _Box("out")), "TOrOp", ("lhs", "rhs", "out"), {}),
    ("exp", lambda: tile_api.exp(_Box("inp"), _Box("out")), "TExpOp", ("inp", "out"), {}),
    ("log", lambda: tile_api.log(_Box("inp"), _Box("out")), "TLogOp", ("inp", "out"), {}),
    ("relu", lambda: tile_api.relu(_Box("inp"), _Box("out")), "TReluOp", ("inp", "out"), {}),
    ("abs", lambda: tile_api.abs(_Box("inp"), _Box("out")), "TAbsOp", ("inp", "out"), {}),
    ("sqrt", lambda: tile_api.sqrt(_Box("inp"), _Box("out")), "TSqrtOp", ("inp", "out"), {}),
    ("rsqrt", lambda: tile_api.rsqrt(_Box("inp"), _Box("out")), "TRsqrtOp", ("inp", "out"), {}),
    ("reciprocal", lambda: tile_api.reciprocal(_Box("inp"), _Box("out")), "TRecipOp", ("inp", "out"), {}),
    ("matmul", lambda: tile_api.matmul(_Box("lhs"), _Box("rhs"), _Box("out")), "TMatmulOp", (None, "lhs", "rhs", "out"), {}),
    ("matmul_bias", lambda: tile_api.matmul_bias(_Box("lhs"), _Box("rhs"), _Box("bias"), _Box("out")), "TMatmulBiasOp", (None, "lhs", "rhs", "bias", "out"), {}),
    ("matmul_acc", lambda: tile_api.matmul_acc(_Box("acc"), _Box("lhs"), _Box("rhs"), _Box("out")), "TMatmulAccOp", (None, "acc", "lhs", "rhs", "out"), {}),
    ("row_sum", lambda: tile_api.row_sum(_Box("src"), _Box("tmp"), _Box("dst")), "TRowSumOp", (), {"src": "src", "tmp": "tmp", "dst": "dst"}),
    ("row_min", lambda: tile_api.row_min(_Box("src"), _Box("tmp"), _Box("dst")), "TRowMinOp", (), {"src": "src", "tmp": "tmp", "dst": "dst"}),
    ("row_max", lambda: tile_api.row_max(_Box("src"), _Box("tmp"), _Box("dst")), "TRowMaxOp", (), {"src": "src", "tmp": "tmp", "dst": "dst"}),
    ("row_expand", lambda: tile_api.row_expand(_Box("src"), _Box("dst")), "TRowExpandOp", (), {"src": "src", "dst": "dst"}),
    ("row_expand_sub", lambda: tile_api.row_expand_sub(_Box("src0"), _Box("src1"), _Box("dst")), "TRowExpandSubOp", (), {"src0": "src0", "src1": "src1", "dst": "dst"}),
    ("row_expand_div", lambda: tile_api.row_expand_div(_Box("src0"), _Box("src1"), _Box("dst")), "TRowExpandDivOp", (), {"src0": "src0", "src1": "src1", "dst": "dst"}),
    ("row_expand_mul", lambda: tile_api.row_expand_mul(_Box("src0"), _Box("src1"), _Box("dst")), "TRowExpandMulOp", (), {"src0": "src0", "src1": "src1", "dst": "dst"}),
    ("col_sum", lambda: tile_api.col_sum(_Box("src"), _Box("tmp"), _Box("dst"), is_binary=False), "TColSumOp", (), {"src": "src", "tmp": "tmp", "dst": "dst", "isBinary": "false"}),
    ("col_min", lambda: tile_api.col_min(_Box("src"), _Box("dst")), "TColMinOp", (), {"src": "src", "dst": "dst"}),
    ("col_max", lambda: tile_api.col_max(_Box("src"), _Box("dst")), "TColMaxOp", (), {"src": "src", "dst": "dst"}),
    ("col_expand", lambda: tile_api.col_expand(_Box("src"), _Box("dst")), "TColExpandOp", (), {"src": "src", "dst": "dst"}),
    ("sort32", lambda: tile_api.sort32(_Box("src"), _Box("dst"), _Box("idx")), "TSort32Op", ("src", "dst", "idx"), {}),
]

SPECIAL_CASES = {"gather", "extract", "mrgsort", "subset"}


def test_tile_export_coverage_is_complete():
    covered = {name for name, *_ in CALL_CASES} | SPECIAL_CASES
    assert covered == set(tile_api.__all__)


def test_call_based_tile_ops_dispatch_to_the_expected_underlying_builders(
    monkeypatch, mlir_ctx
):
    seen = []

    def fake_call(op, *args, **kwargs):
        seen.append(
            (
                op,
                tuple(tile_api._unwrap(arg) for arg in args),
                {name: tile_api._unwrap(value) for name, value in kwargs.items()},
            )
        )

    monkeypatch.setattr(tile_api, "_call", fake_call)

    for _, invoker, expected_op_name, expected_args, expected_kwargs in CALL_CASES:
        invoker()
        op, args, kwargs = seen.pop(0)
        assert op is getattr(tile_api._pto, expected_op_name)
        assert args == expected_args
        assert {key: str(value) for key, value in kwargs.items()} == expected_kwargs


def test_gather_supports_indices_and_mask_pattern(monkeypatch):
    seen = []

    def fake_call(op, *args, **kwargs):
        seen.append(
            (
                op,
                tuple(tile_api._unwrap(arg) for arg in args),
                {name: tile_api._unwrap(value) for name, value in kwargs.items()},
            )
        )

    monkeypatch.setattr(tile_api, "_call", fake_call)
    monkeypatch.setattr(tile_api._pto, "MaskPattern", SimpleNamespace(PAT_ALL="PAT_ALL"))
    monkeypatch.setattr(
        tile_api._pto,
        "MaskPatternAttr",
        SimpleNamespace(get=lambda value: f"mask:{value}"),
    )

    tile_api.gather(_Box("src"), _Box("dst"), indices=_Box("idx"))
    tile_api.gather(_Box("src"), _Box("dst"), mask_pattern="PAT_ALL")

    assert seen[0] == (
        tile_api._pto.TGatherOp,
        ("src", "dst"),
        {"indices": "idx"},
    )
    assert seen[1] == (
        tile_api._pto.TGatherOp,
        ("src", "dst"),
        {"maskPattern": "mask:PAT_ALL"},
    )


def test_extract_unwraps_source_and_indices(monkeypatch):
    seen = {}

    def fake_extract(**kwargs):
        seen.update({name: tile_api._unwrap(value) for name, value in kwargs.items()})

    monkeypatch.setattr(tile_api._pto, "TExtractOp", fake_extract)

    tile_api.extract(_Box("src"), _Box("row"), _Box("col"), _Box("dst"))

    assert seen == {
        "src": "src",
        "indexRow": "row",
        "indexCol": "col",
        "dst": "dst",
    }


def test_mrgsort_casts_block_length_and_wraps_src_dst_lists(monkeypatch, mlir_ctx):
    seen = {}

    def fake_index_cast(dtype, value):
        seen["index_cast"] = (str(dtype), value)
        return SimpleNamespace(result=f"cast:{value}")

    def fake_mrgsort(**kwargs):
        seen["mrgsort"] = kwargs

    monkeypatch.setattr(tile_api._arith, "IndexCastOp", fake_index_cast)
    monkeypatch.setattr(tile_api._pto, "TMrgSortOp", fake_mrgsort)

    tile_api.mrgsort(_Box("src"), _Box("dst"), _Box("block"))

    assert seen["index_cast"][1] == "block"
    assert seen["mrgsort"] == {
        "srcs": ["src"],
        "dsts": ["dst"],
        "blockLen": "cast:block",
    }


def test_subset_unwraps_offsets_and_returns_underlying_result(monkeypatch):
    seen = {}

    def fake_subset(source, offsets, sizes):
        seen["call"] = (source, offsets, sizes)
        return "subset-result"

    monkeypatch.setattr(tile_api._pto, "subset", fake_subset)

    result = tile_api.subset(_Box("src"), [_Box("r0"), _Box("r1")], [32, 64])

    assert result == "subset-result"
    assert seen["call"] == ("src", ["r0", "r1"], [32, 64])
