# Minimal Reproducer: Unwanted `TRESHAPE` for no-addr `alloc_tile`

## Summary

This reproducer highlights an extra `TRESHAPE` emitted in generated C++ when
`pto.alloc_tile` is used **without** `addr` (default build level).

For a semantically equivalent kernel compiled at level3 (with explicit `addr`),
the generated C++ does **not** include this extra `TRESHAPE`.

Known-by-design level gating (`addr` only allowed in `--pto-level=level3`) is
not the issue here; the issue is the additional reshape introduced in default
mode output.

## Reproducer Files

- `ir_builder.py`: emits minimal PTO module with one tile allocation + `tload`
  - no-addr: `python ir_builder.py > noaddr.pto`
  - with-addr: `python ir_builder.py --with-addr > withaddr.pto`
- `compile.sh`: compiles two successful cases:
  - default level (no addr)
  - level3 (with addr)

## How to Run

```bash
cd debug_alloc_tile
bash ./compile.sh
```

## Observed Difference in Generated C++

### A) Default level, no addr

```bash
ptoas noaddr.pto -o noaddr.cpp
```

`noaddr.cpp` contains an extra reshape:

```cpp
Tile<...RowMajor...> v9;
TASSIGN(v9, v8);
Tile<...ColMajor..., SLayout::RowMajor...> v10;
TRESHAPE(v10, v9);   // extra conversion
TLOAD(v10, v13);
```

### B) Level3, with addr

```bash
ptoas --pto-level=level3 withaddr.pto -o withaddr_level3.cpp
```

`withaddr_level3.cpp` does not need reshape:

```cpp
Tile<...ColMajor..., SLayout::RowMajor...> v9;
TASSIGN(v9, v8);
TLOAD(v9, v12);
```

## Why This Is Problematic

- The default-level path introduces additional generated operations
  (`TRESHAPE`) that are absent in the level3 path for equivalent tile usage.
- This makes codegen behavior inconsistent across build levels and can affect
  readability/debuggability and possibly performance-sensitive paths.

## Relevant PTOAS Areas

- `PTOAS/tools/ptoas/ptoas.cpp`
  - Contains known level gating for `alloc_tile.addr` (design behavior).
- `PTOAS/lib/PTO/Transforms/PTOToEmitC.cpp`
  - Lowering path where tile view semantics can emit `TRESHAPE` in default mode.

## Request

Please review why default-level no-addr `alloc_tile` lowering materializes
`TRESHAPE` for this minimal case, and whether codegen can avoid this extra step
when source/destination tile semantics are effectively equivalent for the
generated `TLOAD` usage.

