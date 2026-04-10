"""Auto-discovered compile-level tests: ptoas assembly and bisheng compilation.

These tests **automatically discover** DSL kernel definitions from sibling
``test_*_ir.py`` modules and verify:

1. DSL-generated ``.pto`` can be assembled to C++ by ``ptoas``.
2. The ptoas C++ output + a caller wrapper compiles to ``.so`` via ``bisheng``.

Neither step requires an NPU device — only the CLI tools that ship with the
CANN image.

Convention for auto-discovery
-----------------------------
Modules named ``test_*_ir.py`` in this directory are scanned at collection
time.  Each module that defines a top-level ``meta_data()`` function is
inspected.  Every other public, non-test, non-build function whose parameter
annotations are all plain strings (PTO type aliases like ``"ptr_type"``) is
treated as a DSL kernel.  Each one is wrapped with
``to_ir_module(meta_data=meta_data)(fn)`` to produce a ``.pto`` module, then:

1. The ``.pto`` is assembled to C++ by ``ptoas``.
2. The ptoas C++ output + a caller wrapper compiles to ``.so`` via ``bisheng``.

``--enable-insert-sync`` is enabled automatically unless the generated IR
already contains explicit synchronisation ops (``record_event`` /
``wait_event``).

Raw MLIR ``build*()`` functions in the same modules are **not** picked up
for compilation — they exist solely for IR-equality tests.

To add a new op, create a ``test_<op>_ir.py`` that follows the pattern above.
**No changes to this file are required.**
"""

import importlib.util
import inspect
import pathlib
import sys

import pytest

from ptodsl import JitWrapper, to_ir_module

from conftest import run_ptoas, run_bisheng

_THIS_DIR = pathlib.Path(__file__).parent


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _import_ir_modules():
    """Import every ``test_*_ir.py`` sibling and return ``{stem: module}``."""
    modules = {}
    for path in sorted(_THIS_DIR.glob("test_*_ir.py")):
        stem = path.stem
        # Prefer the already-loaded module (pytest may have imported it).
        if stem in sys.modules:
            modules[stem] = sys.modules[stem]
        else:
            spec = importlib.util.spec_from_file_location(stem, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[stem] = mod
            spec.loader.exec_module(mod)
            modules[stem] = mod
    return modules


def _is_kernel_fn(fn, mod):
    """Heuristic: *fn* is a PTO DSL kernel defined in *mod*."""
    if not inspect.isfunction(fn):
        return False
    # Must be defined in *this* source file (filters out re-exports).
    try:
        fn_file = inspect.getfile(fn)
    except TypeError:
        return False
    if fn_file != getattr(mod, "__file__", None):
        return False
    hints = getattr(fn, "__annotations__", {})
    params = {k: v for k, v in hints.items() if k != "return"}
    if not params:
        return False
    # All parameter annotations must be plain strings (PTO type aliases).
    return all(isinstance(v, str) for v in params.values())


def _has_manual_sync(ir_text: str) -> bool:
    """``True`` when the IR contains explicit sync ops."""
    return "record_event" in ir_text or "wait_event" in ir_text


# ---------------------------------------------------------------------------
# Collect cases at import time
# ---------------------------------------------------------------------------

# Each entry: (case_id, ir_factory, meta_fn, kernel_fn)
_CASES: list[tuple] = []

for _mod_name, _mod in _import_ir_modules().items():
    _short = _mod_name.removeprefix("test_").removesuffix("_ir")

    # -- DSL kernels: meta_data + annotated functions ----------------------
    _meta = getattr(_mod, "meta_data", None)
    if callable(_meta):
        for _name, _obj in inspect.getmembers(_mod, inspect.isfunction):
            if (
                _name.startswith("_")
                or _name.startswith("test_")
                or _name.startswith("build")
                or _name == "meta_data"
            ):
                continue
            if _is_kernel_fn(_obj, _mod):
                _cid = f"{_short}/{_name}"

                # default-arg trick to capture the current loop values
                def _make_dsl_factory(m=_meta, k=_obj):
                    return to_ir_module(meta_data=m)(k)

                _CASES.append((_cid, _make_dsl_factory, _meta, _obj))

# -- Parametrise lists -----------------------------------------------------

_PTOAS_PARAMS = [pytest.param(cid, factory, id=cid) for cid, factory, _, _ in _CASES]

_BISHENG_PARAMS = [
    pytest.param(cid, factory, meta, kern, id=cid)
    for cid, factory, meta, kern in _CASES
    if meta is not None and kern is not None
]


# ---------------------------------------------------------------------------
# Caller-cpp helper (reuses JitWrapper internals)
# ---------------------------------------------------------------------------


def _caller_cpp(fn, meta_data_fn, kernel_cpp_name="kernel.cpp"):
    """Use :class:`JitWrapper` internals to generate a ``caller.cpp`` string."""
    wrapper = JitWrapper(fn, meta_data=meta_data_fn, block_dim=20)
    wrapper._arg_types = wrapper._resolve_runtime_arg_types()
    return wrapper._generate_caller_cpp(kernel_cpp_name)


# ---------------------------------------------------------------------------
# ptoas assembly tests
# ---------------------------------------------------------------------------


@pytest.mark.require_ptoas_cli
@pytest.mark.parametrize("case_id,ir_factory", _PTOAS_PARAMS)
def test_ptoas_assemble(case_id, ir_factory, tmp_path):
    """``ptoas`` can assemble auto-discovered .pto into C++."""
    ir_module = ir_factory()
    ir_text = str(ir_module)
    enable_sync = not _has_manual_sync(ir_text)

    pto_path = tmp_path / "kernel.pto"
    cpp_path = tmp_path / "kernel.cpp"
    pto_path.write_text(ir_text + "\n", encoding="utf-8")

    run_ptoas(pto_path, cpp_path, enable_insert_sync=enable_sync)

    assert cpp_path.exists(), f"ptoas did not produce {cpp_path}"
    content = cpp_path.read_text(encoding="utf-8")
    assert len(content) > 0, "ptoas produced an empty C++ file"


# ---------------------------------------------------------------------------
# bisheng end-to-end compile tests  (ptoas → bisheng)
# ---------------------------------------------------------------------------


@pytest.mark.require_ptoas_cli
@pytest.mark.require_bisheng
@pytest.mark.parametrize("case_id,ir_factory,meta_fn,kernel_fn", _BISHENG_PARAMS)
def test_bisheng_compile(case_id, ir_factory, meta_fn, kernel_fn, tmp_path):
    """Full pipeline: .pto → ptoas → .cpp → bisheng → .so"""
    ir_module = ir_factory()
    ir_text = str(ir_module)
    enable_sync = not _has_manual_sync(ir_text)

    pto_path = tmp_path / "kernel.pto"
    cpp_path = tmp_path / "kernel.cpp"
    caller_path = tmp_path / "caller.cpp"
    so_path = tmp_path / "kernel.so"

    # Step 1: ptoas
    pto_path.write_text(ir_text + "\n", encoding="utf-8")
    run_ptoas(pto_path, cpp_path, enable_insert_sync=enable_sync)

    # Step 2: generate caller.cpp
    caller_code = _caller_cpp(kernel_fn, meta_fn, kernel_cpp_name=cpp_path.name)
    caller_path.write_text(caller_code, encoding="utf-8")

    # Step 3: bisheng
    run_bisheng(caller_path, so_path, cwd=str(tmp_path))

    assert so_path.exists(), f"bisheng did not produce {so_path}"
    assert so_path.stat().st_size > 0, "bisheng produced an empty .so"
