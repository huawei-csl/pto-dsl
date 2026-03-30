# PTO `t*` + `v*` 混合示例

## 一页版表达

```text
Outer PTO tile flow:
  make_tensor_view -> partition_view -> tload -> [vector inner loop] -> tstore

Inner vector loop:
  vlds -> vlds -> vadd -> vsts
```

## PPT 版伪 IR

```mlir
module {
  func.func @vec_add_mixed(
      %a: !pto.ptr<f32>,
      %b: !pto.ptr<f32>,
      %c: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index

    // 1) 先用 PTO tile op 选出一个 32x32 工作块
    %A = pto.make_tensor_view %a, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %B = pto.make_tensor_view %b, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %C = pto.make_tensor_view %c, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>

    %tileA = pto.partition_view %A, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %tileB = pto.partition_view %B, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %tileC = pto.partition_view %C, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

    // 统一记号：!tile 表示 vec-local 32x32 f32 tile_buf
    %bufA = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %bufB = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>
    %bufC = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>

    // 2) tile 级搬运：GM -> local tile
    pto.tload ins(%tileA : !pto.partition_tensor_view<32x32xf32>)
      outs(%bufA : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>)
    pto.tload ins(%tileB : !pto.partition_tensor_view<32x32xf32>)
      outs(%bufB : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>)

    // 3) vector 级计算：在 local tile 内部按 64-lane 分块
    %ptrA = pto.tile_buf_addr %bufA : !pto.tile_buf<...> -> !llvm.ptr<6>
    %ptrB = pto.tile_buf_addr %bufB : !pto.tile_buf<...> -> !llvm.ptr<6>
    %ptrC = pto.tile_buf_addr %bufC : !pto.tile_buf<...> -> !llvm.ptr<6>

    scf.for %i = %c0 to %c1024 step %c64 {
      %va = pto.vlds %ptrA[%i] : !llvm.ptr<6> -> !pto.vreg<64xf32>
      %vb = pto.vlds %ptrB[%i] : !llvm.ptr<6> -> !pto.vreg<64xf32>
      %vc = pto.vadd %va, %vb
        : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>
      pto.vsts %vc, %ptrC[%i] : !pto.vreg<64xf32>, !llvm.ptr<6>
    }

    // 4) tile 级写回：local tile -> GM
    pto.tstore ins(%bufC : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, ...>)
      outs(%tileC : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
```

## 讲解时只强调这两层

- `pto.t*` 负责选 tile 和搬 tile：`make_tensor_view -> partition_view -> tload -> tstore`
- `pto.v*` 负责在 tile 内做向量计算：`vlds -> vadd -> vsts`
