---
name: translate-cpp2py
description: Translate manual PTO-ISA C++ kernels into PTO-DSL Python builders and verification harnesses. Use when converting pto-isa kernel code to ptodsl, generating .pto/.cpp via ptoas, handling manual vs auto sync variants, separating vector vs cube APIs, or adding missing ptodsl API wrappers.
---

# Translate PTO-ISA C++ to PTO-DSL

## Scope

This skill converts a manually written PTO C++ kernel into:
- a **manual-sync** PTO-DSL Python builder (must mirror source C++ behavior),
- an **auto-sync** PTO-DSL variant (same math/control flow, sync removed),
- generated `.pto` and `.cpp`,
- launcher and runtime correctness test scripts.

Primary references are under `references/example_translation`. Only consult long compiler/dialect sources when mapping is missing.

## Required Outputs Per Translation Task

Produce all of the following unless user asks otherwise:
- Python builder for **manual-sync** kernel.
- Python builder for **auto-sync** kernel.
- Compile scripts:
  - manual: `python builder.py > kernel.pto && ptoas kernel.pto -o kernel.cpp`
  - auto: `python builder.py > kernel.pto && ptoas --enable-insert-sync kernel.pto -o kernel.cpp`
- `caller.cpp` kernel launcher with correct ABI and launch geometry.
- `run_*.py` load-and-test script to validate numerical correctness.
- `README.md` with minimal usage commands (compile + run + optional bench), following concise style used in `examples/aot/*/README.md`.

## Non-Negotiable Rules

1. Input C++ is manual-sync by default. Port to manual-sync Python first.
2. Then create auto-sync variant by removing explicit sync APIs and compiling with `--enable-insert-sync`.
3. Preserve ABI exactly: function name, argument order/types, launch contract.
4. Match section type exactly: vector (`__DAV_VEC__`) vs cube (`__DAV_CUBE__`).
5. Prefer compact Python; preserve semantics, not C++ verbosity.
6. If wrapper is missing in `ptodsl/api`, add it instead of forcing awkward translation.
7. First check if the directory `references/example_translation` is empty or contains too few examples,
   If empty, ask for running `scripts/collect_example_translate.py` to generate full Python-C++ mapping examples.


## Translation Workflow

1. **Classify kernel**
   - Determine section: vector vs cube.
   - Determine sync style: manual vs auto (source C++ is manual).
   - Identify core partitioning pattern (block/subblock/batch split).

2. **Rebuild signature + metadata first**
   - Define `meta_data()` with scalar/index/pointer/tensor/subtensor/tile types.
   - Use `@to_ir_module(meta_data=meta_data)`.
   - Keep argument order identical to C++ kernel ABI.

3. **Port runtime control flow**
   - Use `pto.range`, `pto.if_context`, `pto.cond` for runtime logic.
   - Keep all tail guards and truncation branches.

4. **Port data movement + tile math**
   - Build tensors via `pto.as_tensor`.
   - Create subviews with `pto.slice_view`.
   - Allocate tiles with `pto.alloc_tile`.
   - Map load/store/compute ops 1:1 (see mapping rules below).

5. **Handle synchronization**
   - Manual variant: keep explicit event/barrier calls.
   - Auto variant: remove manual sync calls, keep op order, compile with insert-sync pass.

6. **Generate and verify round-trip**
   - Emit `.pto`, compile to `.cpp`, and sanity-check structural equivalence.
   - Build `.so` with `caller.cpp`.
   - Run Python test script against reference (`torch` or equivalent).

## Sync Modes (Must Explain in Every Task)

- **Manual sync mode**
  - Python uses explicit sync APIs in `ptodsl/api/synchronization.py`.
  - Typical APIs: `record_event`, `wait_event`, `record_wait_pair`, `barrier`.
  - Compile with plain `ptoas` (no `--enable-insert-sync`).
  - Use for direct mirroring of manual C++ or for hand-tuned pipelines.

- **Auto sync mode**
  - Remove explicit sync APIs from Python DSL.
  - Compile with `ptoas --enable-insert-sync`.
  - Compiler inserts hazard-handling synchronization.
  - Use for simpler maintainable variant with same algorithmic behavior.

Rule of thumb: one kernel variant should use one sync strategy only.

## Vector vs Cube Section/API Boundaries

- **Vector kernels**
  - Use `with pto.vector_section():`
  - Lowers to `#if defined(__DAV_VEC__)`.
  - Typical ops: elementwise/reduction/vector dataflow (`tile.add/sub/mul/div/...`).

- **Cube kernels**
  - Use `with pto.cube_section():`
  - Lowers to `#if defined(__DAV_CUBE__)`.
  - Typical ops: matrix engines (`tile.matmul`, `tile.matmul_acc`, `tile.matmul_bias`).

- **API surface filtering**
  - Vector-only example: `tile.add` in `ptodsl/api/tile.py`.
  - Cube-only example: `tile.matmul` in `ptodsl/api/tile.py`.
  - Keep agent search narrow: choose section first, then look only at relevant API family.

## Compact Mapping Rules (Python -> C++)

1. `@to_ir_module` function -> emitted `__global__ AICORE void ...`.
2. `PtrType(dtype)` -> C++ GM pointer arg type.
3. `TensorType/SubTensorType` + `as_tensor/slice_view` -> `GlobalTensor` objects/views.
4. `TileBufType(memory_space=...)` + `alloc_tile` -> tile declarations in corresponding memory space.
5. `pto.get_block_idx/get_block_num/get_subblock_idx/get_subblock_num` -> runtime core/subcore intrinsics.
6. `s.const/s.index_cast/s.ceil_div/s.select/min` -> scalar arithmetic + branch/select expressions.
7. `pto.range(...)` -> runtime loop in IR/C++.
8. Python `range(...)` -> build-time unroll/metaprogramming.
9. `pto.if_context(...)` / `pto.cond(...)` -> runtime conditional branches.
10. Python `if` -> build-time branch while constructing IR.
11. `pto.load` / `pto.store` -> load/store tile movement ops.
12. `tile.add/sub/mul/div/relu/exp/...` -> corresponding PTO compute intrinsics.
13. `tile.matmul*` family -> cube matmul intrinsics.
14. Multicore distribution usually maps via:
    - vector core id = `block_idx * subblock_num + subblock_idx` (vector core is 2x than cube core, `subblock_num` equals 2)
    - tiles per core = ceil-div(total tiles, total cores)
    - guarded tail processing for final core(s).
15. Dynamic-shape kernels require explicit bound guards before slicing/loading/storing.

## Runtime Semantics Reminder (Critical)

PTO-DSL is Python tracing, not AST rewriting:
- Python-native `if/for` executes at build time, similar to C++ compile-time metaprogramming or loop unrolling
- Only `pto.range` and `pto.if_context` represent runtime control flow in generated kernel.

Never translate runtime C++ control logic into Python-native `if/range` by mistake.

## Missing API Wrapper Protocol

If required C++ op has no convenient Python wrapper:

1. Add thin wrapper in the right module:
   - tile/instruction ops -> `ptodsl/api/tile.py`
   - general tensor/control helpers -> `ptodsl/api/pto_general.py`
   - sync helpers -> `ptodsl/api/synchronization.py`
2. Re-export through `ptodsl/api/pto.py` when needed.
3. Keep wrapper minimal: pass through to MLIR Python binding op with light argument normalization.

## Escalation Path (Only When Mapping Is Missing)

Check in order:
1. MLIR Python op bindings: `references/ptoas_source/pto.py`
2. Dialect op definitions/contracts: `references/ptoas_source/PTOOps.td`
3. C++ codegen lowering: `references/ptoas_source/PTOToEmitC.cpp`
4. ISA semantics: `references/ptoisa_source/pto-inst.hpp`

If op exists in dialect but not lowered in `PTOToEmitC.cpp`, translation requires PTOAS compiler work (not only DSL wrapper work).
In this case, suggest an issue report to PTOAS project (https://github.com/zhangstevenunity/PTOAS)

## Round-Trip Verification Checklist

- [ ] Manual-sync Python version created first and compiles with plain `ptoas`.
- [ ] Auto-sync variant created and compiles with `--enable-insert-sync`.
- [ ] Generated C++ keeps ABI/section/loop/tail semantics.
- [ ] Launcher `caller.cpp` matches kernel symbol and launch parameters.
- [ ] Test script loads `.so`, runs multiple shapes (including tail/non-divisible cases), compares against trusted reference.
- [ ] If multicore kernel: test cases include shapes not multiples of core count.
- [ ] `README.md` documents the exact local commands to compile and run verification.

## Reference Priority

Use these first:
- `references/example_translation/**` (primary mapping corpus)
- `references/example_translation/fast_hadamard/**` (manual vs auto sync pair)
- `references/example_translation/batch_matmul/**` (cube kernels)
- `examples/aot/elementwise/add_dynamic_multicore/*` (caller/test/build pattern)
- `examples/aot/matmul_optimization_guide/matmul_optim_guide.md` (sync and runtime-control semantics)

Consult `references/ptoas_source/**` and ISA headers only for patterns not covered by examples.
