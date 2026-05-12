# Flash Attention 140 TFLOP/s DSL Builders

This directory has two PTODSL Flash Attention builders:

- `fa_dsl_builder.py`: default `TILE_S1=256` builder.
- `fa_dsl_builder_tile512.py`: experimental `TILE_S1=512` builder.

Both compile scripts write the same runtime artifact:

```text
build_artifacts/fa_dsl.so
```

`run.py` always loads that file. Compile first, then run.

## Build

Default 256-tile kernel:

```bash
bash compile.sh
```

Experimental 512-tile kernel:

```bash
bash compile_tile512.sh
```

The 512-tile builder uses `TILE_S1=512` and `QK_PRELOAD=4`, so it requires
`S1 >= 2048`. The default `run.py` sweep skips `S1=1024` for this builder.

## Run

Run one or more specific sequence lengths:

```bash
python run.py --s1-values 8192
python run.py --s1-values 8192,131072
```

Benchmark PTODSL perf only for one sequence length:

```bash
python run.py --perf-mode 131072
```

## Vector Barrier Removal Experiment

Both compile scripts accept selected generated-C++ vector barrier removals:

```bash
bash compile.sh --remove-vec-barriers line1,line2,...
bash compile_tile512.sh --remove-vec-barriers line1,line2,...
```

This only removes lines containing:

```cpp
pipe_barrier(PIPE_V);
```

The patched C++ is emitted as:

```text
build_artifacts/fa_dsl_patched.cpp
```

The compiled shared object is still:

```text
build_artifacts/fa_dsl.so
```

### Known Useful 256-Tile Variant

Use:

```bash
bash compile.sh --remove-vec-barriers 1264,1267,1272,1275,1279,1282,1311,1313,1316,1320,1322,1325,1328,1330,1333,1362,1364,1367,1371,1373,1376,1379,1381,1384,1390
python run.py --perf-mode 131072
```
