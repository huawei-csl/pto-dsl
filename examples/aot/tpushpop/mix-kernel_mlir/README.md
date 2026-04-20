# TPush / TPop mixed-kernel examples

Small examples of tile FIFO communication between Cube (`AIC`) and Vector (`AIV`).

```bash
python run.py c2v
python run.py c2v_add
python run.py v2c
python run.py bidi
```

`python run.py` defaults to `c2v`.

Files:

- `kernels/` has the Python builders.
- `build_artifacts/` gets generated MLIR, generated C++, and the `.so`.
- `gm_slot_buffer` is the GM backing store for the pipe.
- `caller.cpp` sets the FFTS base before launching the generated kernel.

Core idea:

- `aic_initialize_pipe` / `aiv_initialize_pipe` lower to matching `TPipe<...>` objects.
- `gm_slot_buffer` is the shared GM slot memory used by that `TPipe`.
- `tpush_to_aiv` / `tpush_to_aic` lower to `TPUSH(pipe, tile)`.
- `tpop_from_aic` / `tpop_from_aiv` lower to `TPOP(pipe, tile)`.
- `tfree_from_aic` / `tfree_from_aiv` lower to `TFREE(pipe)` and release the consumed slot.
- `run.py` launches `get_num_cube_cores()` blocks for every mode.
- Inputs and outputs are shaped as `[block_dim, 16, 16]`.
- Each launched block pair gets its own `8 KiB` GM slot region and uses
  `get_block_idx()` plus `addptr` so blocks do not overwrite each other's FIFO
  slots or input/output tiles.

## C2V

Cube sends. Vector receives.

This example launches one Cube/Vector block pair per cube core reported by
`get_num_cube_cores()`. Each block computes `X[block] @ X[block]` on Cube,
sends the accumulator tile to Vector with `split=1`, then each vector subblock
stores one row half to `Y[block]`.

```text
Cube:   load X[block] -> matmul -> tpush_to_aiv split=1
Vector: tpop_from_aic split=1 -> store Y[block] half -> tfree_from_aic split=1
```

Pipe wiring:

- Vector owns the consumer buffer: `reserve_buffer("c2v_fifo", location="VEC")`
- Cube imports it: `import_reserved_buffer("c2v_fifo", peer_func="@vector_kernel")`
- Both sides initialize with `dir_mask = 1`

## C2V Add

Cube sends. Vector receives and doubles.

This example uses the same multicore C2V split as `c2v`, then each vector
subblock doubles its received `8x16` row half before storing it to `Y[block]`.

```text
Cube:   load X[block] -> matmul -> tpush_to_aiv split=1
Vector: tpop_from_aic split=1 -> add -> store Y[block] half -> tfree_from_aic split=1
```

## V2C

Vector sends. Cube receives.

This example launches one Cube/Vector block pair per cube core reported by
`get_num_cube_cores()`. Each vector subblock loads one `8x16` row half from
`X[block]`, sends it to Cube with `split=1`, then Cube receives the reassembled
`16x16` tile and stores it to `Y[block]`.

```text
Vector: load X[block] half -> tpush_to_aic split=1
Cube:   tpop_from_aiv split=1 -> store Y[block] -> tfree_from_aiv split=1
```

Pipe wiring:

- Cube owns the consumer buffer: `reserve_buffer("v2c_fifo", location="MAT")`
- Vector imports it: `import_reserved_buffer("v2c_fifo", peer_func="@cube_kernel")`
- Both sides initialize with `dir_mask = 2`

## BIDI

Both directions are enabled.

This example launches one Cube/Vector block pair per cube core reported by
`get_num_cube_cores()`. Each block sends its own `X[block] @ X[block]` tile
from Cube to Vector. Vector doubles it and sends it back. Cube receives the
returned tile and stores it to `Y[block]`.

```text
Cube:   matmul -> tpush_to_aiv
Vector: tpop_from_aic -> add -> tpush_to_aic -> tfree_from_aic
Cube:   tpop_from_aiv -> store Y -> tfree_from_aiv
```

Pipe wiring:

- Vector reserves `c2v_fifo`; Cube imports it
- Cube reserves `v2c_fifo`; Vector imports it
- Both sides initialize with `dir_mask = 3`

For `dir_mask = 3`, allocate FIFO backing for both directions. Each launched
block pair gets its own `8 KiB` GM slot region:

```text
2 directions * 4 slots * 1024 bytes = 8192 bytes per block
```

`run.py` allocates `get_num_cube_cores() * 8192` bytes of GM FIFO backing for
each mode.
