<div align="center">

# PTO-DSL
Pythonic interface and JIT compiler for [PTO-ISA](https://gitcode.com/cann/pto-isa)
</div>

PTO-DSL provides a programming abstraction similar to [cuTile](https://docs.nvidia.com/cuda/cutile-python/), but native to [NPU](https://www.hiascend.com/).

**Key features:**
- Automatic software pipelining without [manual synchronization](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0179.html)
- Easily interface with [torch-npu](https://gitcode.com/ascend/pytorch)
- Lightweight, open-source compiler stack using [PTO Assembler](https://github.com/zhangstevenunity/PTOAS)

**Compare to other kernel programming frameworks** (e.g. [tilelang-ascend](https://github.com/tile-ai/tilelang-ascend), [triton-ascend](https://gitcode.com/Ascend/triton-ascend), and [catlass](https://gitcode.com/cann/catlass)):
- PTO-DSL aims for **low-level, explicit, NPU-native primitives** that can match the performance of **programming in [hardware intrinsics](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/cceintrinsicapi/cceapi_0001.html)**, filling the gap of a [CuteDSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)-like low-level Python programming for NPU.

## Environment

See [docker](./docker)

## Installation

```bash
pip install -e ./ptodsl
```

## Usage

See [examples](./examples) and [tests](./tests)

## Contribute

See [contribute_guide.md](./contribute_guide.md)
