# Minimal Reproducer: `alloc_tile` + `addr` Level Coupling

## Summary

This reproducer demonstrates that `pto.alloc_tile` with `addr` is accepted only at `--pto-level=level3`, while `pto.alloc_tile` without `addr` is rejected at level3.

In practice, this forces users to choose between:

- default level (no `addr` allowed), and
- level3 (explicit `addr` required).

## Reproducer Files

- `ir_builder.py`: emits a minimal PTO module with exactly one `pto.alloc_tile`
  - no `addr` mode: `python ir_builder.py > noaddr.pto`
  - with `addr` mode: `python ir_builder.py --with-addr > withaddr.pto`
- `compile.sh`: runs four compile cases and prints exit codes.

## How to Run

```bash
cd debug_alloc_tile
bash ./compile.sh
```

## Expected/Observed Results

### Case A: default level + no addr

```bash
ptoas noaddr.pto -o noaddr.cpp
```

Expected: success  
Observed: success

### Case B: default level + with addr

```bash
ptoas withaddr.pto -o withaddr_default.cpp
```

Expected: fail  
Observed: fail, with:

> unexpected 'addr' operand: only supported when --pto-level=level3

### Case C: level3 + with addr

```bash
ptoas --pto-level=level3 withaddr.pto -o withaddr_level3.cpp
```

Expected: success  
Observed: success

### Case D: level3 + no addr

```bash
ptoas --pto-level=level3 noaddr.pto -o noaddr_level3.cpp
```

Expected: fail  
Observed: fail, with:

> requires 'addr' operand when --pto-level=level3

## Where This Is Enforced

The behavior is explicitly enforced in `ptoas`:

- `PTOAS/tools/ptoas/ptoas.cpp`
  - In level3:
    - rejects missing `addr` on `pto::AllocTileOp`
  - In non-level3:
    - rejects present `addr` on `pto::AllocTileOp`

Relevant logic (roughly around lines 604-626):

- `if (effectiveLevel == PTOBuildLevel::Level3) { ... requires 'addr' ... }`
- `else { ... unexpected 'addr' ... }`

## Why This Matters

For users trying to move from explicit-address flow (`level3`) to default-level flow, this strict split can expose behavior differences in generated C++ (for example, additional intermediate shape/view adaptation paths depending on lowering), but there is no mixed mode to keep explicit allocation intent while testing non-level3 compilation.

## Suggested Discussion Points for PTOAS

1. Whether `addr` handling should remain strictly level-gated, or allow a compatibility mode.
2. Whether a warning-based migration mode is desirable (instead of hard error).
3. Whether downstream lowering behavior can be made more consistent between level3 and non-level3 for equivalent tile semantics.

