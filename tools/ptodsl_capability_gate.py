#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from pathlib import Path

from mlir import ir
from mlir.dialects import pto as raw_pto

from ptodsl import pto


ROOT = Path(__file__).resolve().parents[1]
PTO_ISA_ROOT = ROOT.parent / "pto-isa"
PTOAS_ROOT = ROOT.parent / "PTOAS"
PTOAS_HIDDEN = {"TileBufType"}
PTODSL_EXTRA = {
    "TileType",
    "TileConfig",
    "section",
    "fillpad",
    "fillpad_expand",
}
REQUIRED_DTYPES = {
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float8_e4m3fn",
    "float8_e5m2",
}
REQUIRED_PTOAS_TYPE_METHODS = {
    "TileType": {"to_buffer"},
}


def _run_manifest_script(script: Path) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def build_report():
    ptoas_script = PTOAS_ROOT / "tools" / "gen_pto_capability_manifest.py"
    isa_script = PTO_ISA_ROOT / "scripts" / "gen_pto_isa_capability_manifest.py"
    ptoas_manifest = _run_manifest_script(ptoas_script) if ptoas_script.exists() else None
    isa_manifest = _run_manifest_script(isa_script) if isa_script.exists() else None

    with ir.Context() as ctx:
        raw_pto.register_dialect(ctx, load=True)
        if ptoas_manifest is not None:
            ptoas_exports = set(ptoas_manifest.get("python_symbols", [])) - PTOAS_HIDDEN
        else:
            ptoas_exports = {name for name in dir(raw_pto) if not name.startswith("_")} - PTOAS_HIDDEN
        ptodsl_exports = set(pto.__all__)
        missing_ptodsl_symbols = sorted(ptoas_exports - ptodsl_exports)
        missing_dtypes = sorted(name for name in REQUIRED_DTYPES if not hasattr(pto, name))
        if isa_manifest is not None:
            isa_dtype_coverage = isa_manifest.get("frontend_dtypes", {})
        else:
            isa_dtype_coverage = {}

        report = {
            "ptoas_export_count": len(ptoas_exports),
            "ptodsl_export_count": len(ptodsl_exports),
            "missing_ptodsl_symbols": missing_ptodsl_symbols,
            "missing_dtypes": missing_dtypes,
            "tile_namespace_removed": not hasattr(pto, "TileBufType"),
            "isa_dtype_coverage": isa_dtype_coverage,
            "ptoas_manifest": str(ptoas_script) if ptoas_manifest is not None else None,
            "pto_isa_manifest": str(isa_script) if isa_manifest is not None else None,
            "ptoas_canonical_frontend_type": (
                ptoas_manifest.get("canonical_frontend_type") if ptoas_manifest is not None else None
            ),
            "missing_ptoas_type_methods": {
                type_name: sorted(required - set(ptoas_manifest.get("python_type_methods", {}).get(type_name, [])))
                for type_name, required in REQUIRED_PTOAS_TYPE_METHODS.items()
                if ptoas_manifest is not None and required - set(ptoas_manifest.get("python_type_methods", {}).get(type_name, []))
            },
            "extra_ptodsl_exports": sorted(
                name for name in ptodsl_exports
                if name not in ptoas_exports and name not in REQUIRED_DTYPES and name not in PTODSL_EXTRA
            ),
        }
    report["ok"] = (
        not report["missing_ptodsl_symbols"]
        and not report["missing_dtypes"]
        and report["tile_namespace_removed"]
        and (not report["isa_dtype_coverage"] or all(report["isa_dtype_coverage"].values()))
        and report["ptoas_canonical_frontend_type"] in (None, "TileType")
        and not report["missing_ptoas_type_methods"]
    )
    return report


def main():
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
