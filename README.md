<div align="center">

# PTO-DSL
Pythonic interface and JIT compiler for [PTO-ISA](https://gitcode.com/cann/pto-isa)
</div>

**Key features:**
- Easy interfacing with [torch-npu](https://gitcode.com/ascend/pytorch)
- Automatic software pipelining without [manual synchronization](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0179.html)
- Explicit control across memory hierarchy
- Simple and lightweight compiler stack without over-engineering

PTO-DSL provides a programming abstraction layer similar to [Cute DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) and [cuTile](https://docs.nvidia.com/cuda/cutile-python/), but native to [NPU](https://www.hiascend.com/).

## Environment

See [docker](./docker)

## Installation

```bash
pip install -e ./ptodsl
```

## Usage

See [examples](./examples) and [tests](./tests)
