# Sinkhorn K=4 (fp16) — AOT demo

Self-contained PTODSL builder, `ptoas` auto-sync emit, bisheng `.so`, and NPU tests (DeepSeek-style `sinkhorn_normalize`).

## Prerequisites

- `ptoas`, `bisheng`, Ascend NPU + `torch_npu`
- PTO-ISA headers: set `PTO_LIB_PATH` to a [pto-isa](https://github.com/zhangstevenunity/pto-isa) checkout, or use an environment where `/sources/pto-isa/include` exists

## Compile

```bash
cd examples/aot/sinkhorn_demo
./compile.sh
```

Override repo root if needed: `PTO_DSL_ROOT=/path/to/pto-dsl ./compile.sh`.

## Tests (NPU)

From the **pto-dsl** repository root:

```bash
python3 -m pytest examples/aot/sinkhorn_demo/test_sinkhorn.py -v --npu=0
```

Or from this directory:

```bash
cd examples/aot/sinkhorn_demo
python3 -m pytest test_sinkhorn.py -v --npu=0
```

Pick a free device (`npu-smi info`) and pass `--npu=<id>` or `--npu=npu:<id>`.

On first import, `jit_util_sinkhorn` can also regenerate MLIR/C++ and compile the `.so` if you run Python from this directory.
