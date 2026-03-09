#!/usr/bin/env python3
"""
Minimal repro for alloc_tile behavior across ptoas build levels.

Usage:
  python ir_builder.py             > noaddr.pto
  python ir_builder.py --with-addr > withaddr.pto
"""

import argparse

from mlir.dialects import arith
from mlir.ir import IntegerType
from ptodsl import pto, to_ir_module


def build(with_addr: bool):
    def meta_data():
        dtype = pto.float16
        ptr_type = pto.PtrType(dtype)
        i64 = IntegerType.get_signless(64)

        # One tile type is enough to trigger the alloc_tile checks/paths.
        tile_buf = pto.TileBufType(
            shape=[128, 128],
            dtype=dtype,
            memory_space="MAT",
        )

        return {
            "ptr_type": ptr_type,
            "i64": i64,
            "tile_buf": tile_buf,
        }

    @to_ir_module(meta_data=meta_data)
    def kernel(a_ptr: "ptr_type") -> None:
        # Keep ptr alive to avoid accidental pruning in very aggressive pipelines.
        _ = a_ptr
        with pto.cube_section():
            if with_addr:
                c0_i64 = arith.ConstantOp(i64, 0).result
                _tile = pto.alloc_tile(tile_buf, addr=c0_i64)
            else:
                _tile = pto.alloc_tile(tile_buf)

    return kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-addr",
        action="store_true",
        help="Emit pto.alloc_tile with explicit addr operand",
    )
    args = parser.parse_args()

    print(build(with_addr=args.with_addr))


if __name__ == "__main__":
    main()

