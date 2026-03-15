# Bug Report: PTO-DSL cannot express `ACC -> MAT` `TMOV`

## Summary

When translating a manual PTO kernel pattern that uses feedback moves like:

- `TMOV(Y_l1_tile, c_l0_tile)` (`ACC -> MAT`)

the equivalent PTO-DSL path fails, while manual C++ works.

Observed failures:

1. **IR verify failure** (when `ACC` and `MAT` dtypes differ):
   - `error: unknown: 'pto.tmov' op expects src/dst to have the same element type`
2. **C++ compile failure** (after forcing same dtype):
   - `TMov: Invalid TileType` for `TMOV(Mat, Acc)`

---

## Environment

- Platform: Ascend `dav-2201`
- CANN: `8.5.0`
- Toolchain: `ptoas`, `bisheng`
- Repro directory: `pto-dsl/examples/aot/fast_inverse/bug_report`

---

## Reproducer Files (local, self-contained)

- Failing PTO-DSL reproducer: `repro_fail_tmov_acc_to_mat.py`
- Failing raw-MLIR reproducer (no ptodsl wrappers): `reproducer_raw_api.py`
- Working PTO-DSL workaround: `repro_workaround_spill_acc_to_mat.py`
- Working manual C++ reproducer: `repro_manual_acc_to_mat.cpp`

Compile helpers:

- `compile_bug.sh` (expected failure)
- `compile_workaround.sh` (expected success for workaround path)
- `compile_cpp.sh` (expected success for manual C++ path)

---

## How To Run

```bash
cd pto-dsl/examples/aot/fast_inverse/bug_report

# 1) Expected failure at IR stage
bash ./compile_bug.sh
# (optional raw API path)
python3 ./reproducer_raw_api.py > /tmp/reproducer_raw_api.pto

# 2) Expected success via workaround (ACC->GM->MAT)
bash ./compile_workaround.sh

# 3) Expected success for manual C++ reproducer
bash ./compile_cpp.sh
```

---

## Failing PTO-DSL Reproducer

<details>
<summary><code>repro_fail_tmov_acc_to_mat.py</code> (click to expand)</summary>

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
def repro_fail_tmov_acc_to_mat(out_ptr: "ptr_out", in_ptr: "ptr_in", n_i32: "i32") -> None:
    with pto.cube_section():
        c0 = const(0)
        c1 = const(1)
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
        tile.mov(tc, tmat)  # ACC -> MAT (expected failure)
        pto.store(tc, sy)


if __name__ == "__main__":
    print(repro_fail_tmov_acc_to_mat)
```

</details>

---

## Working PTO-DSL Workaround

<details>
<summary><code>repro_workaround_spill_acc_to_mat.py</code> (click to expand)</summary>

```python
from ptodsl import pto, tile, to_ir_module
from ptodsl import scalar as s

const = s.const

# Same idea as project workaround:
# ACC -> GM (store) -> MAT (load), instead of ACC -> MAT TMOV.
```

</details>

Key workaround logic:

<details>
<summary>Workaround snippet</summary>

```python
def spill_acc_to_mat(dst_l1):
    sync("MATMUL", "STORE_ACC")
    pto.store(c_l0, sv_out)
    sync("STORE_ACC", "LOAD")
    pto.load(sv_out, dst_l1)
    sync("LOAD", "MOV_M2L")
```

</details>

---

## Working Manual C++ Reproducer

<details>
<summary><code>repro_manual_acc_to_mat.cpp</code> (click to expand)</summary>

```cpp
#define MEMORY_BASE
#include <pto/pto-inst.hpp>

using namespace pto;

extern "C" __global__ AICORE void repro_manual_acc_to_mat(
    __gm__ float *out_ptr, __gm__ half *in_ptr, int32_t n_i32) {
#if (__CHECK_FEATURE_AT_PRECOMPILE) || \
    (__CCE_AICORE__ == 220 && defined(__DAV_C220_CUBE__))
  const uint32_t n = static_cast<uint32_t>(n_i32);

  using TensorShapeIn = TileShape2D<half, 128, 128, Layout::ND>;
  using TensorStridesIn = BaseShape2D<half, 128, 128, Layout::ND>;
  using TensorShapeOut = TileShape2D<float, 128, 128, Layout::ND>;
  using TensorStridesOut = BaseShape2D<float, 128, 128, Layout::ND>;

  using GlobalTensorIn =
      GlobalTensor<half, TensorShapeIn, TensorStridesIn, Layout::ND>;
  using GlobalTensorOut =
      GlobalTensor<float, TensorShapeOut, TensorStridesOut, Layout::ND>;

  using TileL1 = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128,
                      128, SLayout::RowMajor, 512>;
  using TileL0A = TileLeft<half, 128, 128>;
  using TileL0B = TileRight<half, 128, 128>;
  using TileL0C = TileAcc<float, 128, 128>;

  GlobalTensorIn in_gm(in_ptr);
  GlobalTensorOut out_gm(out_ptr);

  TileL1 y_l1;
  TileL0A a_l0;
  TileL0B b_l0;
  TileL0C c_l0;

  TASSIGN(y_l1, 0x0);
  TASSIGN(a_l0, 0x0);
  TASSIGN(b_l0, 0x0);
  TASSIGN(c_l0, 0x0);

  y_l1.SetValidRow(n);
  y_l1.SetValidCol(n);
  a_l0.SetValidRow(n);
  a_l0.SetValidCol(n);
  b_l0.SetValidRow(n);
  b_l0.SetValidCol(n);
  c_l0.SetValidRow(n);
  c_l0.SetValidCol(n);

  TLOAD(y_l1, in_gm);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

  TMOV(a_l0, y_l1);
  TMOV(b_l0, y_l1);
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

  TMATMUL(c_l0, a_l0, b_l0);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

  TMOV(y_l1, c_l0);  // ACC(float) -> MAT(half): known manual working path
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

  TMATMUL(c_l0, a_l0, b_l0);
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
  TSTORE(out_gm, c_l0);
#endif
}
```

</details>

Manual C++ critical pattern:

<details>
<summary>Critical `ACC -> MAT` pattern</summary>

```cpp
TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
TMOV(Y_l1_tile, c_l0_tile);  // ACC -> MAT
```

</details>

---

## Impact

- Blocks faithful translation of kernels requiring `ACC -> MAT` feedback.
- Forces GM spill/reload workaround, which can alter performance and dynamic-shape behavior.

## Request

Please support `ACC -> MAT` move semantics in PTO-DSL lowering when backend/manual PTO supports corresponding `TMOV` usage, or provide an official DSL-level equivalent that preserves semantics without workaround-specific behavior changes.
