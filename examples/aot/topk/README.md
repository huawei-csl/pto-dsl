# TopK (AOT, dynamic n_rows, float32)

Finds the top-K largest elements per row of a 2-D `[N_ROWS × N_COLS]` float32
matrix using a TSORT32 → TMRGSORT → TGATHER pipeline on the NPU vector engine.

`N_ROWS` is a **runtime** argument — a single compiled `.so` handles any row
count without recompilation.  `N_COLS`, `TOPK`, and `BLOCK_DIM` are
compile-time constants because they govern tile buffer sizes and the number of
merge-sort passes, which must be statically known by the hardware.

## Parameters

| Symbol           | Kind         | Default | Meaning                              |
|------------------|:------------:|--------:|--------------------------------------|
| `N_ROWS`         | **runtime**  | any     | rows in the input matrix             |
| `N_COLS`         | compile-time | 512     | input elements per row               |
| `TOPK`           | compile-time | 256     | top-k output count per row           |
| `BLOCK_DIM`      | compile-time | 24      | number of NPU compute blocks         |
| `SORT_BLOCK_LEN` | compile-time | 32      | TSORT32 sorts in blocks of this many |

Valid `N_COLS` values (with `SORT_BLOCK_LEN=32`):

| `N_COLS` | `SORT_COLS` | Merge passes |
|---------:|------------:|:------------:|
| 128      | 256         | 1            |
| 512      | 1024        | 2            |
| 2048     | 4096        | 3            |

## Pipeline (per row)

```text
input row [1 x N_COLS]  -->  TSORT32          -->  sort buffer [1 x 2*N_COLS]
                                                    (interleaved score/idx pairs)
                             TMRGSORT x passes -->  fully sorted [1 x 2*N_COLS]
                             TMOV (gather window, valid=[1 x 2*TOPK])
                             TGATHER P0101     -->  tb_scores  [1 x TOPK]  float32
                             TGATHER P1010     -->  tb_indices [1 x TOPK]  uint32
```

The gather-window tile has `valid_shape=[1, 2*TOPK]`, which limits TGATHER
to exactly `TOPK` outputs even when `TOPK < N_COLS`.

## Usage

Compile all configs and validate all 11 test cases:

```bash
python ./run_topk.py
```

To compile a single config manually or skip recompilation:

```text
# compile one config: N_COLS TOPK BLOCK_DIM
bash ./compile.sh 512 256 24

# skip recompilation if .so files already exist
python ./run_topk.py --no-compile
```

## Files

| File              | Purpose                                                    |
|-------------------|------------------------------------------------------------|
| `topk_builder.py` | PTO-DSL builder – emits MLIR for a given `(N_COLS, TOPK)` |
| `caller.py`       | Generates `caller.cpp` with `int32_t n_rows` at call time  |
| `compile.sh`      | End-to-end build: PTO → MLIR → C++ → `.so`                |
| `run_topk.py`     | Validates 11 configs against `torch.topk`                  |

Generated build artifacts (gitignored):

| Artifact                         | Created by   |
|----------------------------------|--------------|
| `caller.cpp`                     | `compile.sh` |
| `topk_c<N_COLS>_k<TOPK>.pto`    | `compile.sh` |
| `topk_c<N_COLS>_k<TOPK>.cpp`    | `compile.sh` |
| `topk_c<N_COLS>_k<TOPK>_lib.so` | `compile.sh` |
