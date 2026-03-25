````markdown
# TopK (AOT, static shape, float32)

Finds the top-K largest elements per row of a 2-D matrix using a
TSORT32 → TMRGSORT → TGATHER pipeline on the NPU vector engine.

## Parameters

| Symbol        | Value | Meaning                                  |
|---------------|------:|------------------------------------------|
| `N_ROWS`      | 4800  | total rows in the input matrix           |
| `N_COLS`      | 512   | input elements per row                   |
| `TOPK`        | 256   | top-k output count per row               |
| `BLOCK_DIM`   | 24    | number of NPU compute blocks             |
| `SORT_BLOCK_LEN` | 32 | TSORT32 sorts in blocks of this many     |

## Pipeline (per row)

```
input row [1 × 512]  →  TSORT32       →  sort buffer [1 × 1024]
                                           (interleaved score/idx pairs)
                         TMRGSORT ×2   →  fully sorted  [1 × 1024]
                         TGATHER P0101 →  tb_scores  [1 × 256]  float32
                         TMOV + TGATHER P1010
                                       →  tb_indices [1 × 256]  uint32
```

## Usage

```bash
bash ./compile.sh   # generate MLIR, compile with bisheng → topk_float32_lib.so
python ./run_topk.py
```

## Files

| File                    | Purpose                                          |
|-------------------------|--------------------------------------------------|
| `topk_builder.py`       | PTO-DSL builder – generates the MLIR kernel      |
| `caller.cpp`            | C++ wrapper – `call_topk_float32(stream, …)`     |
| `compile.sh`            | End-to-end build: PTO → MLIR → C++ → `.so`       |
| `run_topk.py`           | Runs the kernel and validates against `torch.topk`|
| `topk_float32.pto`      | Generated MLIR (after `compile.sh`)              |
| `topk_float32.cpp`      | Generated C++ (after `compile.sh`)               |
| `topk_float32_lib.so`   | Compiled shared library (after `compile.sh`)     |
````
