<div align="center">

# PTO-DSL
Pythonic interface and JIT compiler for [PTO-ISA](https://github.com/PTO-ISA/pto-isa)
</div>

PTO-DSL provides a programming abstraction similar to [cuTile](https://docs.nvidia.com/cuda/cutile-python/), but native to [NPU](https://www.hiascend.com/).

**Key features:**
- Automatic software pipelining without [manual synchronization](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0179.html)
- Easily interface with [torch-npu](https://gitcode.com/ascend/pytorch)
- Lightweight, open-source compiler stack using [PTO Assembler](https://github.com/zhangstevenunity/PTOAS)

## Installation

See [docker/README.md](./docker/README.md) for full reproducible dependencies on NPU.

Then, install this lightweight DSL package itself:

```bash
# install latest commit
pip install git+https://github.com/huawei-csl/pto-dsl.git

# or stable tag
pip install git+https://github.com/huawei-csl/pto-dsl.git@0.1.0
```

For in-place development:

```bash
git clone https://github.com/huawei-csl/pto-dsl.git
cd pto-dsl
pip install -e .
```

## Usage

See [examples](./examples) and [tests](./tests)

Preferred frontend style keeps the existing low-level ops available, but adds a thinner
object-centric layer for common tensor and tile flows:

```python
from ptodsl import pto, tile


def vec_add(src0: "ptr_t", src1: "ptr_t", dst: "ptr_t", rows: "index_t", cols: "index_t"):
    x = pto.make_tensor(src0, shape=[rows, cols], dtype=pto.float32)
    y = pto.make_tensor(src1, shape=[rows, cols], dtype=pto.float32)
    z = pto.make_tensor(dst, shape=[rows, cols], dtype=pto.float32)

    x_tile = x.slice([0, 0], [32, 32])
    y_tile = y.slice([0, 0], [32, 32])
    z_tile = z.slice([0, 0], [32, 32])

    with pto.vector_section():
        tile_buf = pto.make_tile_buffer(pto.float32, [32, 32], space="VEC")
        lhs = tile_buf.alloc()
        rhs = tile_buf.alloc()
        out = tile_buf.alloc()
        lhs.load_from(x_tile)
        rhs.load_from(y_tile)
        tile.add(lhs, rhs, out)
        out.store_to(z_tile)
```

The lower-level `PtrType`, `TensorType`, `SubTensorType`, `TileBufType`, `as_tensor`,
`slice_view`, and `alloc_tile` APIs remain supported for cases where explicit control is
preferred.

## Contribute

See [contribute_guide.md](./contribute_guide.md)

## Compare to other frameworks

PTO-DSL aims for **low-level, explicit, NPU-native primitives** that can match the performance of **programming in [hardware intrinsics](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/cceintrinsicapi/cceapi_0001.html)**. Compared to other (also very good) kernel programming frameworks, it has a bit different scope by design:
- vs [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend): tilelang can also [use PTO-ISA as codegen backend](https://github.com/tile-ai/tilelang-ascend/blob/76553755da078479a7f60cce9c5f0e9a24d0008b/src/target/codegen_ascend_pto.cc). PTO-DSL intentionally exposes lower-level control, for example L2 swizzling is one-liner `T.use_swizzle` in tilelang, but is a user-defined custom function in PTO-DSL -- see this [matmul optimization example](examples/aot/matmul_optimization_guide/matmul_optim_guide.md). Once PTO-DSL is more stabilized, it might serve as a component like the [CuteDSL backend for tilelang](https://github.com/tile-ai/tilelang/blob/v0.1.8/src/target/codegen_cutedsl.cc).
- vs [triton-ascend](https://gitcode.com/Ascend/triton-ascend): Both frameworks automate software pipelining based on some MLIR dialects for NPU. PTO-DSL exposes more NPU-native memory hierarchy such as `L0`/`L1`/`UB`. Also, `pto.load`/`pto.store` always maps to native efficient DMA instructions, while `tl.load`/`tl.store` tries to do GPU-style memory coalescing.
- vs [Catlass](https://gitcode.com/cann/catlass): Catlass provides expert-optimized template collections, while PTO-DSL is more like the [CuteDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) layer of Cutlass, offering explicit low-level primitives.
- vs [PyPTO](https://gitcode.com/cann/pypto): PyPTO is a full [MPMD](https://en.wikipedia.org/wiki/Flynn%27s_taxonomy#Multiple_programs,_multiple_data_streams_(MPMD)) dynamic runtime stack, which also [uses PTO-ISA as lowest-level primitive](https://gitcode.com/cann/pypto/tree/r0.1.1/framework/src/interface/tileop). PyPTO's Tensor API abstraction is closer to PyTorch/JAX level, while a PTO-DSL kernel is still [SPMD](https://en.wikipedia.org/wiki/Single_program,_multiple_data) and is closer to CuTile/CuteDSL level.
