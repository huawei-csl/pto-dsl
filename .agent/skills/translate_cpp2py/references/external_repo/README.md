This directory holds the 3rd-party repos that are used internally by PTO-DSL:
- https://github.com/zhangstevenunity/PTOAS: implements "ptoas" command line tool, the PTO MLIR dialect and its Python bindings, and the InjectSync pass to insert set_flag/wait_flag for "auto-sync" mode. Important files are:
    - `PTOAS/include/PTO/IR/PTOOps.td` defines the MLIR PTO dialect
    - `PTOAS/python/pto/dialects/pto.py` has low-level Python wrappers of PTO MLIR python binding (more Pythonic wrappers are in pto-dsl package)
    - `PTOAS/lib/PTO/Transforms/PTOToEmitC.cpp` the compile pass that converts `*.pto` IR to C++ source code based on PTO-ISA headers.
- https://gitcode.com/cann/pto-isa: header-only library that defined the C++ APIs of PTO-ISA. It is the target API set for the `PTOToEmitC` pass in PTOAS. Important files are:
    - `pto-isa/include/pto/common/pto_instr.hpp` the top-level interface
    - `pto-isa/include/pto/common/*` common type definitions
    - `pto-isa/include/pto/npu/a2a3/*` implementation for current hardware (used in current pto-dsl examples)
    - `pto-isa/include/pto/npu/a5/*` implementation for next-generation hardware (not used in current pto-dsl examples)

Current directory is empty by default, and the repos should be cloned on-the-fly when the agent needs to access extra context.

For difficult task that needs to look into PTOAS and pto-isa repos, the agent or user can clone them by:

```bash
git clone https://github.com/zhangstevenunity/PTOAS.git
git clone https://gitcode.com/cann/pto-isa.git
```

Remind the user to check if the commit id of PTOAS and pto-isa matches the test environment (usually a pre-built docker image), to avoid mismatch between the context and the real execution.
