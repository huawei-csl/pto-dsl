import subprocess

from mlir.dialects import func, pto as _pto
from mlir.ir import (
    Attribute,
    Context,
    FlatSymbolRefAttr,
    InsertionPoint,
    Location,
    Module,
    Operation,
    UnitAttr,
)

from ptodsl import pto, to_ir_module


def meta_data():
    dtype = pto.float32
    ptr_ty = pto.PtrType(dtype)
    return {"ptr_ty": ptr_ty}


@to_ir_module(meta_data=meta_data)
def single_kernel(arg0: "ptr_ty") -> None:
    pass


@to_ir_module(meta_data=meta_data, module=True)
def multi_kernel_module():
    @pto.func(kernel="vector")
    def worker(arg0: "ptr_ty") -> None:
        pass

    @pto.func
    def entry(arg0: "ptr_ty") -> None:
        pto.call(worker, arg0)


def build_single_verbose():
    with Context() as ctx, Location.unknown():
        _pto.register_dialect(ctx, load=True)
        module = Module.create()
        ptr_ty = _pto.PtrType.get(pto.float32)
        fn_ty = func.FunctionType.get([ptr_ty], [])

        with InsertionPoint(module.body):
            fn = func.FuncOp("single_kernel", fn_ty)
            entry = fn.add_entry_block()

        with InsertionPoint(entry):
            func.ReturnOp([])

        module.operation.verify()
        return module


def build_multi_verbose():
    with Context() as ctx, Location.unknown():
        _pto.register_dialect(ctx, load=True)
        module = Module.create()
        ptr_ty = _pto.PtrType.get(pto.float32)
        fn_ty = func.FunctionType.get([ptr_ty], [])

        with InsertionPoint(module.body):
            worker = func.FuncOp("worker", fn_ty)
            entry = func.FuncOp("entry", fn_ty)

        worker.operation.attributes["pto.kernel_kind"] = Attribute.parse(
            "#pto.kernel_kind<vector>"
        )
        entry.operation.attributes["pto.entry"] = UnitAttr.get(ctx)

        with InsertionPoint(worker.add_entry_block()):
            func.ReturnOp([])

        entry_block = entry.add_entry_block()
        with InsertionPoint(entry_block):
            arg0 = entry_block.arguments[0]
            Operation.create(
                "func.call",
                operands=[arg0],
                attributes={"callee": FlatSymbolRefAttr.get("worker")},
            )
            func.ReturnOp([])

        module.operation.verify()
        return module


def test_old_single_function_builder_matches_raw_mlir():
    assert str(single_kernel) == str(build_single_verbose())


def test_new_multi_function_builder_matches_raw_mlir():
    assert str(multi_kernel_module) == str(build_multi_verbose())


def test_multi_function_module_compiles_with_ptoas(tmp_path):
    pto_path = tmp_path / "multi_kernel_module.pto"
    cpp_path = tmp_path / "multi_kernel_module.cpp"
    pto_path.write_text(str(multi_kernel_module), encoding="utf-8")

    subprocess.run(
        [
            "ptoas",
            "--enable-insert-sync",
            str(pto_path),
            "-o",
            str(cpp_path),
        ],
        check=True,
    )
