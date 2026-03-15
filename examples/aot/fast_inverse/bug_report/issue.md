# Bug Report: PTO-DSL cannot express `ACC -> MAT` `TMOV`

## Summary

When translating `tri_inv_trick` from manual PTO C++ to PTO-DSL Python, the original kernel needs feedback moves like:

- `TMOV(Y_l1_tile, c_l0_tile)` (`ACC -> MAT`)

The manual C++ kernel compiles and runs, but the equivalent PTO-DSL translation fails in two ways:

1. **IR verify failure** (when `ACC` and `MAT` dtypes differ):
   - `error: unknown: 'pto.tmov' op expects src/dst to have the same element type`
2. **C++ compile failure** (even after forcing same dtype):
   - `TMov: Invalid TileType` for `TMOV(Mat, Acc)`

So PTO-DSL currently cannot represent a legal/manual PTO pattern used by this kernel.

---

## Environment

- Platform: Ascend `dav-2201`
- CANN: `8.5.0`
- Compiler tools: `ptoas`, `bisheng`
- Project paths:
  - DSL example: `pto-dsl/examples/aot/fast_inverse`
  - Working manual kernel: `pto-kernels/csrc/kernel/kernel_tri_inv_trick.cpp`

---

## Minimal Reproducer (Fails in PTO-DSL)

Save as `repro_fail_tmov_acc_to_mat.py`:

```python
from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

def meta_data():
    in_dtype = pto.float16
    out_dtype = pto.float32
    i32 = pto.int32
    ptr_in = pto.PtrType(in_dtype)
    ptr_out = pto.PtrType(out_dtype)
    tv_in = pto.TensorType(rank=2, dtype=in_dtype)
    tv_out = pto.TensorType(rank=2, dtype=out_dtype)
    st_in = pto.SubTensorType(shape=[128, 128], dtype=in_dtype)
    st_out = pto.SubTensorType(shape=[128, 128], dtype=out_dtype)
    mat = pto.TileBufType(shape=[128, 128], valid_shape=[-1, -1], dtype=in_dtype, memory_space="MAT")
    left = pto.TileBufType(shape=[128, 128], valid_shape=[-1, -1], dtype=in_dtype, memory_space="LEFT")
    right = pto.TileBufType(shape=[128, 128], valid_shape=[-1, -1], dtype=in_dtype, memory_space="RIGHT")
    acc = pto.TileBufType(shape=[128, 128], valid_shape=[-1, -1], dtype=out_dtype, memory_space="ACC")
    return {
        "ptr_in": ptr_in, "ptr_out": ptr_out, "i32": i32,
        "tv_in": tv_in, "tv_out": tv_out, "st_in": st_in, "st_out": st_out,
        "mat": mat, "left": left, "right": right, "acc": acc,
    }

@to_ir_module(meta_data=meta_data)
def repro(out_ptr: "ptr_out", in_ptr: "ptr_in", n_i32: "i32") -> None:
    with pto.cube_section():
        c0 = const(0); c1 = const(1)
        n = s.index_cast(n_i32)
        x = pto.as_tensor(tv_in, ptr=in_ptr, shape=[n, n], strides=[n, c1])
        y = pto.as_tensor(tv_out, ptr=out_ptr, shape=[n, n], strides=[n, c1])
        sx = pto.slice_view(st_in, source=x, offsets=[c0, c0], sizes=[n, n])
        sy = pto.slice_view(st_out, source=y, offsets=[c0, c0], sizes=[n, n])
        tmat = pto.alloc_tile(mat, valid_row=n, valid_col=n)
        ta = pto.alloc_tile(left, valid_row=n, valid_col=n)
        tb = pto.alloc_tile(right, valid_row=n, valid_col=n)
        tc = pto.alloc_tile(acc, valid_row=n, valid_col=n)
        pto.load(sx, tmat)
        tile.mov(tmat, ta)
        tile.mov(tmat, tb)
        tile.matmul(ta, tb, tc)
        tile.mov(tc, tmat)  # ACC -> MAT (fails)
        pto.store(tc, sy)

if __name__ == "__main__":
    print(repro)
```

Run:

```bash
python3 repro_fail_tmov_acc_to_mat.py > /tmp/repro.pto
```

Expected: IR emission succeeds.  
Actual: fails with `'pto.tmov' op expects src/dst to have the same element type`.

If dtype is changed to make `ACC` same dtype as `MAT`, `ptoas` then fails in C++ with:

- `TMov: Invalid TileType` for `TMOV(Mat, Acc)`

---

## Working PTO-DSL Workaround (Current Project)

Workaround in `pto-dsl/examples/aot/fast_inverse/inverse_builder.py`:

```python
# TMOV does not support ACC as source tile on this backend.
# Use ACC->GM->MAT as a legal feedback path.
def spill_acc_to_mat(dst_l1):
    sync("MATMUL", "STORE_ACC")
    pto.store(c_l0, sv_out)
    sync("STORE_ACC", "LOAD")
    pto.load(sv_out, dst_l1)
    sync("LOAD", "MOV_M2L")
```

This avoids `ACC -> MAT` direct move and allows build/runtime to proceed.

---

## Working Manual C++ Reference

Manual kernel (`pto-kernels/csrc/kernel/kernel_tri_inv_trick.cpp`) contains:

```cpp
TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains M^2
set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

TMOV(Y_l1_tile, c_l0_tile);  // Y_l1 now contains M^2
```

This compiles and works in the original project path.

---

## Compile Script Used

Current script (`pto-dsl/examples/aot/fast_inverse/compile.sh`):

```bash
#!/usr/bin/env bash
set -euo pipefail

rm -f \
    inverse_auto_sync.pto inverse_manual_sync.pto \
    inverse_auto_sync.cpp inverse_manual_sync.cpp \
    inverse_auto_sync_lib.so inverse_manual_sync_lib.so

python ./inverse_builder.py > ./inverse_auto_sync.pto
ptoas --enable-insert-sync ./inverse_auto_sync.pto -o ./inverse_auto_sync.cpp

python ./inverse_builder.py --manual-sync > ./inverse_manual_sync.pto
ptoas ./inverse_manual_sync.pto -o ./inverse_manual_sync.cpp
```

---

## Impact

- Blocks faithful translation of kernels that require `ACC -> MAT` feedback.
- Forces extra GM spill/reload workaround, which changes performance and dynamic-shape behavior.

## Request

Please support `ACC -> MAT` move semantics in PTO-DSL lowering when backend/manual PTO supports corresponding `TMOV` usage in this kernel class, or provide an official DSL op/sequence that preserves equivalent semantics without invalidating dynamic-shape behavior.
