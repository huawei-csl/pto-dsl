# PTO-DSL: Pythonic interface and JIT compiler for [PTO-ISA](https://gitcode.com/cann/pto-isa)

Key features:
- Easily interfacing with [torch-npu](https://gitcode.com/ascend/pytorch)
- Automatic software pipelining without [manual synchronization](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0179.html)
- Explicit control across memory hierarchy

## Environment

See [docker](./docker)

## Installation

```bash
pip install -e ./ptodsl
```

## Usage

See [examples](./examples) and [tests](./tests)
