module {
  func.func @cube_kernel(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f16>, %arg2: !pto.ptr<f16>, %arg3: !pto.ptr<f16>) attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c512 = arith.constant 512 : index
    %c8192 = arith.constant 8192 : index
    %c16 = arith.constant 16 : index
    %c96 = arith.constant 96 : index
    %0 = pto.get_block_num
    %1 = arith.index_cast %0 : i64 to index
    %2 = pto.get_block_idx
    %3 = arith.index_cast %2 : i64 to index
    %4 = arith.divsi %c96, %1 : index
    %5 = arith.remsi %c96, %1 : index
    %6 = arith.addi %4, %c1 : index
    %7 = arith.muli %3, %6 : index
    %8 = arith.addi %4, %c1 : index
    %9 = arith.muli %5, %8 : index
    %10 = arith.subi %3, %5 : index
    %11 = arith.muli %10, %4 : index
    %12 = arith.addi %9, %11 : index
    %13 = arith.cmpi slt, %3, %5 : index
    %14 = arith.select %13, %7, %12 : index
    %15 = arith.cmpi slt, %3, %5 : index
    %16 = arith.addi %4, %c1 : index
    %17 = arith.select %15, %16, %4 : index
    %18 = arith.addi %14, %17 : index
    %c229376 = arith.constant 229376 : index
    %19 = arith.muli %3, %c229376 : index
    %20 = pto.addptr %arg0, %19 : <f32> -> <f32>
    %c0_0 = arith.constant 0 : index
    %21 = pto.addptr %20, %c0_0 : <f32> -> <f32>
    %c131072 = arith.constant 131072 : index
    %22 = pto.addptr %20, %c131072 : <f32> -> <f32>
    %c163840 = arith.constant 163840 : index
    %23 = pto.addptr %20, %c163840 : <f32> -> <f32>
    %24 = pto.import_reserved_buffer{name = "fa_qk_c2v_fifo", peer_func = @vector_kernel} -> i32
    %25 = pto.initialize_l2g2l_pipe{dir_mask = 1, slot_size = 65536, slot_num = 8, local_slot_num = 1} (%21 : !pto.ptr<f32>, %24 : i32) -> !pto.pipe
    %26 = pto.import_reserved_buffer{name = "fa_pv_c2v_fifo", peer_func = @vector_kernel} -> i32
    %27 = pto.initialize_l2g2l_pipe{dir_mask = 1, slot_size = 16384, slot_num = 8, local_slot_num = 1} (%22 : !pto.ptr<f32>, %26 : i32) -> !pto.pipe
    %28 = pto.reserve_buffer{name = "fa_p_v2c_fifo", size = 262144, location = <mat>, auto = false, base = 327680} -> i32
    %c0_i32 = arith.constant 0 : i32
    pto.aic_initialize_pipe {id = 30, dir_mask = 2, slot_size = 32768, nosplit = false}(gm_slot_buffer = %23 : !pto.ptr<f32>, c2v_consumer_buf = %c0_i32 : i32, v2c_consumer_buf = %28 : i32)
    %c0_i64 = arith.constant 0 : i64
    %c0_i64_1 = arith.constant 0 : i64
    %29 = pto.alloc_tile addr = %c0_i64_1 : !pto.tile_buf<mat, 32x128xf16, blayout=col_major, slayout=row_major>
    %c0_i64_2 = arith.constant 0 : i64
    %30 = pto.alloc_tile addr = %c0_i64_2 : !pto.tile_buf<left, 32x128xf16, slayout=row_major>
    %c8192_i64 = arith.constant 8192 : i64
    %31 = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<mat, 128x512xf16, slayout=col_major>
    %32 = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<right, 128x512xf16, slayout=col_major>
    %c0_i64_3 = arith.constant 0 : i64
    %33 = pto.alloc_tile addr = %c0_i64_3 : !pto.tile_buf<acc, 32x512xf32, blayout=col_major, slayout=row_major, fractal=1024>
    %c139264_i64 = arith.constant 139264 : i64
    %34 = pto.alloc_tile addr = %c139264_i64 : !pto.tile_buf<mat, 32x512xf16, blayout=col_major, slayout=row_major>
    %c8192_i64_4 = arith.constant 8192 : i64
    %35 = pto.alloc_tile addr = %c8192_i64_4 : !pto.tile_buf<left, 32x512xf16, slayout=row_major>
    %c172032_i64 = arith.constant 172032 : i64
    %36 = pto.alloc_tile addr = %c172032_i64 : !pto.tile_buf<mat, 512x128xf16, blayout=col_major, slayout=row_major>
    %37 = pto.alloc_tile addr = %c0_i64 : !pto.tile_buf<right, 512x128xf16, slayout=col_major>
    %c65536_i64 = arith.constant 65536 : i64
    %38 = pto.alloc_tile addr = %c65536_i64 : !pto.tile_buf<acc, 32x128xf32, blayout=col_major, slayout=row_major, fractal=1024>
    %c3072 = arith.constant 3072 : index
    %39 = pto.make_tensor_view %arg1, shape = [%c3072, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xf16>
    %40 = pto.make_tensor_view %arg2, shape = [%c128, %c8192], strides = [%c1, %c128] : !pto.tensor_view<?x?xf16>
    %41 = pto.make_tensor_view %arg3, shape = [%c8192, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xf16>
    scf.for %arg4 = %14 to %18 step %c1 {
      %42 = arith.muli %arg4, %c32 : index
      %43 = pto.partition_view %39, offsets = [%42, %c0], sizes = [%c32, %c128] : !pto.tensor_view<?x?xf16>
      pto.tload ins(%43 : !pto.partition_tensor_view<32x128xf16>) outs(%29 : !pto.tile_buf<mat, 32x128xf16, blayout=col_major, slayout=row_major>)
      pto.tmov ins(%29 : !pto.tile_buf<mat, 32x128xf16, blayout=col_major, slayout=row_major>) outs(%30 : !pto.tile_buf<left, 32x128xf16, slayout=row_major>)
      %c0_5 = arith.constant 0 : index
      %44 = pto.partition_view %40, offsets = [%c0, %c0_5], sizes = [%c128, %c512] : !pto.tensor_view<?x?xf16>
      pto.tload ins(%44 : !pto.partition_tensor_view<128x512xf16>) outs(%31 : !pto.tile_buf<mat, 128x512xf16, slayout=col_major>)
      pto.tmov ins(%31 : !pto.tile_buf<mat, 128x512xf16, slayout=col_major>) outs(%32 : !pto.tile_buf<right, 128x512xf16, slayout=col_major>)
      pto.tmatmul ins(%30, %32 : !pto.tile_buf<left, 32x128xf16, slayout=row_major>, !pto.tile_buf<right, 128x512xf16, slayout=col_major>) outs(%33 : !pto.tile_buf<acc, 32x512xf32, blayout=col_major, slayout=row_major, fractal=1024>)
      pto.tpush(%33, %25 : !pto.tile_buf<acc, 32x512xf32, blayout=col_major, slayout=row_major, fractal=1024>, !pto.pipe) {split = 1}
      %c512_6 = arith.constant 512 : index
      %45 = pto.partition_view %40, offsets = [%c0, %c512_6], sizes = [%c128, %c512] : !pto.tensor_view<?x?xf16>
      pto.tload ins(%45 : !pto.partition_tensor_view<128x512xf16>) outs(%31 : !pto.tile_buf<mat, 128x512xf16, slayout=col_major>)
      pto.tmov ins(%31 : !pto.tile_buf<mat, 128x512xf16, slayout=col_major>) outs(%32 : !pto.tile_buf<right, 128x512xf16, slayout=col_major>)
      pto.tmatmul ins(%30, %32 : !pto.tile_buf<left, 32x128xf16, slayout=row_major>, !pto.tile_buf<right, 128x512xf16, slayout=col_major>) outs(%33 : !pto.tile_buf<acc, 32x512xf32, blayout=col_major, slayout=row_major, fractal=1024>)
      pto.tpush(%33, %25 : !pto.tile_buf<acc, 32x512xf32, blayout=col_major, slayout=row_major, fractal=1024>, !pto.pipe) {split = 1}
      %46 = pto.partition_view %41, offsets = [%c0, %c0], sizes = [%c512, %c128] : !pto.tensor_view<?x?xf16>
      pto.tload ins(%46 : !pto.partition_tensor_view<512x128xf16>) outs(%36 : !pto.tile_buf<mat, 512x128xf16, blayout=col_major, slayout=row_major>)
      %c7 = arith.constant 7 : index
      scf.for %arg5 = %c0 to %c7 step %c1 {
        %50 = arith.muli %arg5, %c2 : index
        %c2_7 = arith.constant 2 : index
        %51 = arith.addi %50, %c2_7 : index
        %52 = arith.muli %51, %c512 : index
        %53 = pto.partition_view %40, offsets = [%c0, %52], sizes = [%c128, %c512] : !pto.tensor_view<?x?xf16>
        pto.tload ins(%53 : !pto.partition_tensor_view<128x512xf16>) outs(%31 : !pto.tile_buf<mat, 128x512xf16, slayout=col_major>)
        %54 = pto.tpop_from_aiv {id = 30, split = 1} -> !pto.tile_buf<mat, 32x512xf16, blayout=col_major, slayout=row_major>
        pto.tmov ins(%54 : !pto.tile_buf<mat, 32x512xf16, blayout=col_major, slayout=row_major>) outs(%35 : !pto.tile_buf<left, 32x512xf16, slayout=row_major>)
        pto.tfree_from_aiv {id = 30, split = 1}
        pto.tmov ins(%36 : !pto.tile_buf<mat, 512x128xf16, blayout=col_major, slayout=row_major>) outs(%37 : !pto.tile_buf<right, 512x128xf16, slayout=col_major>)
        %55 = arith.addi %50, %c1 : index
        %56 = arith.muli %55, %c512 : index
        %57 = pto.partition_view %41, offsets = [%56, %c0], sizes = [%c512, %c128] : !pto.tensor_view<?x?xf16>
        pto.tload ins(%57 : !pto.partition_tensor_view<512x128xf16>) outs(%36 : !pto.tile_buf<mat, 512x128xf16, blayout=col_major, slayout=row_major>)
        pto.tmatmul ins(%35, %37 : !pto.tile_buf<left, 32x512xf16, slayout=row_major>, !pto.tile_buf<right, 512x128xf16, slayout=col_major>) outs(%38 : !pto.tile_buf<acc, 32x128xf32, blayout=col_major, slayout=row_major, fractal=1024>)
        pto.tpush(%38, %27 : !pto.tile_buf<acc, 32x128xf32, blayout=col_major, slayout=row_major, fractal=1024>, !pto.pipe) {split = 1}
        pto.tmov ins(%31 : !pto.tile_buf<mat, 128x512xf16, slayout=col_major>) outs(%32 : !pto.tile_buf<right, 128x512xf16, slayout=col_major>)
        pto.tmatmul ins(%30, %32 : !pto.tile_buf<left, 32x128xf16, slayout=row_major>, !pto.tile_buf<right, 128x512xf16, slayout=col_major>) outs(%33 : !pto.tile_buf<acc, 32x512xf32, blayout=col_major, slayout=row_major, fractal=1024>)
        pto.tpush(%33, %25 : !pto.tile_buf<acc, 32x512xf32, blayout=col_major, slayout=row_major, fractal=1024>, !pto.pipe) {split = 1}
        %58 = arith.muli %arg5, %c2 : index
        %59 = arith.addi %58, %c1 : index
        %c2_8 = arith.constant 2 : index
        %60 = arith.addi %59, %c2_8 : index
        %61 = arith.muli %60, %c512 : index
        %62 = pto.partition_view %40, offsets = [%c0, %61], sizes = [%c128, %c512] : !pto.tensor_view<?x?xf16>
        pto.tload ins(%62 : !pto.partition_tensor_view<128x512xf16>) outs(%31 : !pto.tile_buf<mat, 128x512xf16, slayout=col_major>)
        %63 = pto.tpop_from_aiv {id = 30, split = 1} -> !pto.tile_buf<mat, 32x512xf16, blayout=col_major, slayout=row_major>
        pto.tmov ins(%63 : !pto.tile_buf<mat, 32x512xf16, blayout=col_major, slayout=row_major>) outs(%35 : !pto.tile_buf<left, 32x512xf16, slayout=row_major>)
        pto.tfree_from_aiv {id = 30, split = 1}
        pto.tmov ins(%36 : !pto.tile_buf<mat, 512x128xf16, blayout=col_major, slayout=row_major>) outs(%37 : !pto.tile_buf<right, 512x128xf16, slayout=col_major>)
        %64 = arith.addi %59, %c1 : index
        %65 = arith.muli %64, %c512 : index
        %66 = pto.partition_view %41, offsets = [%65, %c0], sizes = [%c512, %c128] : !pto.tensor_view<?x?xf16>
        pto.tload ins(%66 : !pto.partition_tensor_view<512x128xf16>) outs(%36 : !pto.tile_buf<mat, 512x128xf16, blayout=col_major, slayout=row_major>)
        pto.tmatmul ins(%35, %37 : !pto.tile_buf<left, 32x512xf16, slayout=row_major>, !pto.tile_buf<right, 512x128xf16, slayout=col_major>) outs(%38 : !pto.tile_buf<acc, 32x128xf32, blayout=col_major, slayout=row_major, fractal=1024>)
        pto.tpush(%38, %27 : !pto.tile_buf<acc, 32x128xf32, blayout=col_major, slayout=row_major, fractal=1024>, !pto.pipe) {split = 1}
        pto.tmov ins(%31 : !pto.tile_buf<mat, 128x512xf16, slayout=col_major>) outs(%32 : !pto.tile_buf<right, 128x512xf16, slayout=col_major>)
        pto.tmatmul ins(%30, %32 : !pto.tile_buf<left, 32x128xf16, slayout=row_major>, !pto.tile_buf<right, 128x512xf16, slayout=col_major>) outs(%33 : !pto.tile_buf<acc, 32x512xf32, blayout=col_major, slayout=row_major, fractal=1024>)
        pto.tpush(%33, %25 : !pto.tile_buf<acc, 32x512xf32, blayout=col_major, slayout=row_major, fractal=1024>, !pto.pipe) {split = 1}
      }
      %47 = pto.tpop_from_aiv {id = 30, split = 1} -> !pto.tile_buf<mat, 32x512xf16, blayout=col_major, slayout=row_major>
      pto.tmov ins(%47 : !pto.tile_buf<mat, 32x512xf16, blayout=col_major, slayout=row_major>) outs(%35 : !pto.tile_buf<left, 32x512xf16, slayout=row_major>)
      pto.tfree_from_aiv {id = 30, split = 1}
      pto.tmov ins(%36 : !pto.tile_buf<mat, 512x128xf16, blayout=col_major, slayout=row_major>) outs(%37 : !pto.tile_buf<right, 512x128xf16, slayout=col_major>)
      %c7680 = arith.constant 7680 : index
      %48 = pto.partition_view %41, offsets = [%c7680, %c0], sizes = [%c512, %c128] : !pto.tensor_view<?x?xf16>
      pto.tload ins(%48 : !pto.partition_tensor_view<512x128xf16>) outs(%36 : !pto.tile_buf<mat, 512x128xf16, blayout=col_major, slayout=row_major>)
      pto.tmatmul ins(%35, %37 : !pto.tile_buf<left, 32x512xf16, slayout=row_major>, !pto.tile_buf<right, 512x128xf16, slayout=col_major>) outs(%38 : !pto.tile_buf<acc, 32x128xf32, blayout=col_major, slayout=row_major, fractal=1024>)
      pto.tpush(%38, %27 : !pto.tile_buf<acc, 32x128xf32, blayout=col_major, slayout=row_major, fractal=1024>, !pto.pipe) {split = 1}
      %49 = pto.tpop_from_aiv {id = 30, split = 1} -> !pto.tile_buf<mat, 32x512xf16, blayout=col_major, slayout=row_major>
      pto.tmov ins(%49 : !pto.tile_buf<mat, 32x512xf16, blayout=col_major, slayout=row_major>) outs(%35 : !pto.tile_buf<left, 32x512xf16, slayout=row_major>)
      pto.tfree_from_aiv {id = 30, split = 1}
      pto.tmov ins(%36 : !pto.tile_buf<mat, 512x128xf16, blayout=col_major, slayout=row_major>) outs(%37 : !pto.tile_buf<right, 512x128xf16, slayout=col_major>)
      pto.tmatmul ins(%35, %37 : !pto.tile_buf<left, 32x512xf16, slayout=row_major>, !pto.tile_buf<right, 512x128xf16, slayout=col_major>) outs(%38 : !pto.tile_buf<acc, 32x128xf32, blayout=col_major, slayout=row_major, fractal=1024>)
      pto.tpush(%38, %27 : !pto.tile_buf<acc, 32x128xf32, blayout=col_major, slayout=row_major, fractal=1024>, !pto.pipe) {split = 1}
    }
    return
  }
  func.func @vector_kernel(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c16_0 = arith.constant 16 : index
    %c96 = arith.constant 96 : index
    %0 = pto.get_block_num
    %1 = arith.index_cast %0 : i64 to index
    %2 = pto.get_block_idx
    %3 = arith.index_cast %2 : i64 to index
    %4 = arith.divsi %c96, %1 : index
    %5 = arith.remsi %c96, %1 : index
    %6 = arith.addi %4, %c1 : index
    %7 = arith.muli %3, %6 : index
    %8 = arith.addi %4, %c1 : index
    %9 = arith.muli %5, %8 : index
    %10 = arith.subi %3, %5 : index
    %11 = arith.muli %10, %4 : index
    %12 = arith.addi %9, %11 : index
    %13 = arith.cmpi slt, %3, %5 : index
    %14 = arith.select %13, %7, %12 : index
    %15 = arith.cmpi slt, %3, %5 : index
    %16 = arith.addi %4, %c1 : index
    %17 = arith.select %15, %16, %4 : index
    %18 = arith.addi %14, %17 : index
    %c229376 = arith.constant 229376 : index
    %19 = arith.muli %3, %c229376 : index
    %20 = pto.addptr %arg0, %19 : <f32> -> <f32>
    %c0_1 = arith.constant 0 : index
    %21 = pto.addptr %20, %c0_1 : <f32> -> <f32>
    %c131072 = arith.constant 131072 : index
    %22 = pto.addptr %20, %c131072 : <f32> -> <f32>
    %c163840 = arith.constant 163840 : index
    %23 = pto.addptr %20, %c163840 : <f32> -> <f32>
    %24 = pto.reserve_buffer{name = "fa_qk_c2v_fifo", size = 65536, location = <vec>, auto = false, base = 0} -> i32
    %25 = pto.initialize_l2g2l_pipe{dir_mask = 1, slot_size = 65536, slot_num = 8, local_slot_num = 1} (%21 : !pto.ptr<f32>, %24 : i32) -> !pto.pipe
    %26 = pto.reserve_buffer{name = "fa_pv_c2v_fifo", size = 16384, location = <vec>, auto = false, base = 65536} -> i32
    %27 = pto.initialize_l2g2l_pipe{dir_mask = 1, slot_size = 16384, slot_num = 8, local_slot_num = 1} (%22 : !pto.ptr<f32>, %26 : i32) -> !pto.pipe
    %28 = pto.import_reserved_buffer{name = "fa_p_v2c_fifo", peer_func = @cube_kernel} -> i32
    %c0_i32 = arith.constant 0 : i32
    pto.aiv_initialize_pipe {id = 30, dir_mask = 2, slot_size = 32768, nosplit = false}(gm_slot_buffer = %23 : !pto.ptr<f32>, c2v_consumer_buf = %c0_i32 : i32, v2c_consumer_buf = %28 : i32)
    %29 = pto.get_subblock_idx
    %30 = arith.index_cast %29 : i64 to index
    %31 = arith.muli %30, %c16 : index
    %c114688_i64 = arith.constant 114688 : i64
    %32 = pto.alloc_tile addr = %c114688_i64 : !pto.tile_buf<vec, 16x512xf32>
    %c147456_i64 = arith.constant 147456 : i64
    %33 = pto.alloc_tile addr = %c147456_i64 : !pto.tile_buf<vec, 16x512xf32>
    %c163840_i64 = arith.constant 163840 : i64
    %34 = pto.alloc_tile addr = %c163840_i64 : !pto.tile_buf<vec, 16x512xf16>
    %c180224_i64 = arith.constant 180224 : i64
    %35 = pto.alloc_tile addr = %c180224_i64 : !pto.tile_buf<vec, 16x128xf32>
    %c188416_i64 = arith.constant 188416 : i64
    %36 = pto.alloc_tile addr = %c188416_i64 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>
    %c188480_i64 = arith.constant 188480 : i64
    %37 = pto.alloc_tile addr = %c188480_i64 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>
    %c188544_i64 = arith.constant 188544 : i64
    %38 = pto.alloc_tile addr = %c188544_i64 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>
    %c188608_i64 = arith.constant 188608 : i64
    %39 = pto.alloc_tile addr = %c188608_i64 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>
    %c188672_i64 = arith.constant 188672 : i64
    %40 = pto.alloc_tile addr = %c188672_i64 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>
    %c188736_i64 = arith.constant 188736 : i64
    %41 = pto.alloc_tile addr = %c188736_i64 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>
    %cst = arith.constant 0.0883883461 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %c3072 = arith.constant 3072 : index
    %42 = pto.make_tensor_view %arg1, shape = [%c3072, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xf32>
    scf.for %arg2 = %14 to %18 step %c1 {
      %43 = arith.muli %arg2, %c32 : index
      %c81920_i64 = arith.constant 81920 : i64
      %44 = pto.alloc_tile addr = %c81920_i64 : !pto.tile_buf<vec, 16x512xf32>
      pto.tpop(%44, %25 : !pto.tile_buf<vec, 16x512xf32>, !pto.pipe) {split = 1}
      pto.tmuls ins(%44, %cst : !pto.tile_buf<vec, 16x512xf32>, f32) outs(%44 : !pto.tile_buf<vec, 16x512xf32>)
      pto.trowmax ins(%44, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
      %45 = pto.treshape %37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %46 = pto.treshape %36 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %47 = pto.treshape %40 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %48 = pto.treshape %38 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %49 = pto.treshape %39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      pto.trowexpandsub ins(%44, %37 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
      pto.tmuls ins(%45, %cst_2 : !pto.tile_buf<vec, 1x16xf32>, f32) outs(%46 : !pto.tile_buf<vec, 1x16xf32>)
      pto.texp ins(%33 : !pto.tile_buf<vec, 16x512xf32>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
      pto.trowsum ins(%33, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%38 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
      pto.tcvt ins(%33 {rmode = #pto<round_mode CAST_RINT>} : !pto.tile_buf<vec, 16x512xf32>) outs(%34 : !pto.tile_buf<vec, 16x512xf16>)
      pto.tpush_to_aic(%34 : !pto.tile_buf<vec, 16x512xf16>) {id = 30, split = 1}
      pto.tfree(%25 : !pto.pipe) {split = 1}
      %c81920_i64_3 = arith.constant 81920 : i64
      %50 = pto.alloc_tile addr = %c81920_i64_3 : !pto.tile_buf<vec, 16x512xf32>
      pto.tpop(%50, %25 : !pto.tile_buf<vec, 16x512xf32>, !pto.pipe) {split = 1}
      pto.tmuls ins(%50, %cst : !pto.tile_buf<vec, 16x512xf32>, f32) outs(%50 : !pto.tile_buf<vec, 16x512xf32>)
      pto.trowmax ins(%50, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
      %51 = pto.treshape %37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %52 = pto.treshape %36 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %53 = pto.treshape %41 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %54 = pto.treshape %38 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %55 = pto.treshape %39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      pto.tmax ins(%51, %52 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%51 : !pto.tile_buf<vec, 1x16xf32>)
      pto.tsub ins(%52, %51 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%53 : !pto.tile_buf<vec, 1x16xf32>)
      pto.tmuls ins(%51, %cst_2 : !pto.tile_buf<vec, 1x16xf32>, f32) outs(%52 : !pto.tile_buf<vec, 1x16xf32>)
      pto.trowexpandsub ins(%50, %37 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
      pto.texp ins(%53 : !pto.tile_buf<vec, 1x16xf32>) outs(%53 : !pto.tile_buf<vec, 1x16xf32>)
      pto.texp ins(%33 : !pto.tile_buf<vec, 16x512xf32>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
      pto.tmul ins(%54, %53 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%54 : !pto.tile_buf<vec, 1x16xf32>)
      pto.trowsum ins(%33, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
      pto.tadd ins(%54, %55 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%54 : !pto.tile_buf<vec, 1x16xf32>)
      pto.tcvt ins(%33 {rmode = #pto<round_mode CAST_RINT>} : !pto.tile_buf<vec, 16x512xf32>) outs(%34 : !pto.tile_buf<vec, 16x512xf16>)
      pto.tpush_to_aic(%34 : !pto.tile_buf<vec, 16x512xf16>) {id = 30, split = 1}
      pto.tfree(%25 : !pto.pipe) {split = 1}
      %c114688_i64_4 = arith.constant 114688 : i64
      %56 = pto.alloc_tile addr = %c114688_i64_4 : !pto.tile_buf<vec, 16x128xf32>
      pto.tpop(%56, %27 : !pto.tile_buf<vec, 16x128xf32>, !pto.pipe) {split = 1}
      pto.tmov ins(%56 : !pto.tile_buf<vec, 16x128xf32>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
      pto.tfree(%27 : !pto.pipe) {split = 1}
      %c81920_i64_5 = arith.constant 81920 : i64
      %57 = pto.alloc_tile addr = %c81920_i64_5 : !pto.tile_buf<vec, 16x512xf32>
      pto.tpop(%57, %25 : !pto.tile_buf<vec, 16x512xf32>, !pto.pipe) {split = 1}
      pto.tmuls ins(%57, %cst : !pto.tile_buf<vec, 16x512xf32>, f32) outs(%57 : !pto.tile_buf<vec, 16x512xf32>)
      pto.trowmax ins(%57, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
      %58 = pto.treshape %37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %59 = pto.treshape %36 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %60 = pto.treshape %40 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %61 = pto.treshape %38 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %62 = pto.treshape %39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      pto.tmax ins(%58, %59 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%58 : !pto.tile_buf<vec, 1x16xf32>)
      pto.tsub ins(%59, %58 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%60 : !pto.tile_buf<vec, 1x16xf32>)
      pto.tmuls ins(%58, %cst_2 : !pto.tile_buf<vec, 1x16xf32>, f32) outs(%59 : !pto.tile_buf<vec, 1x16xf32>)
      pto.trowexpandsub ins(%57, %37 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
      pto.texp ins(%60 : !pto.tile_buf<vec, 1x16xf32>) outs(%60 : !pto.tile_buf<vec, 1x16xf32>)
      pto.texp ins(%33 : !pto.tile_buf<vec, 16x512xf32>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
      pto.tmul ins(%61, %60 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%61 : !pto.tile_buf<vec, 1x16xf32>)
      pto.trowsum ins(%33, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
      pto.tadd ins(%61, %62 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%61 : !pto.tile_buf<vec, 1x16xf32>)
      pto.tcvt ins(%33 {rmode = #pto<round_mode CAST_RINT>} : !pto.tile_buf<vec, 16x512xf32>) outs(%34 : !pto.tile_buf<vec, 16x512xf16>)
      pto.tpush_to_aic(%34 : !pto.tile_buf<vec, 16x512xf16>) {id = 30, split = 1}
      pto.tfree(%25 : !pto.pipe) {split = 1}
      %c114688_i64_6 = arith.constant 114688 : i64
      %63 = pto.alloc_tile addr = %c114688_i64_6 : !pto.tile_buf<vec, 16x128xf32>
      pto.tpop(%63, %27 : !pto.tile_buf<vec, 16x128xf32>, !pto.pipe) {split = 1}
      pto.trowexpandmul ins(%35, %41 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
      pto.tadd ins(%35, %63 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x128xf32>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
      pto.tfree(%27 : !pto.pipe) {split = 1}
      %c81920_i64_7 = arith.constant 81920 : i64
      %64 = pto.alloc_tile addr = %c81920_i64_7 : !pto.tile_buf<vec, 16x512xf32>
      pto.tpop(%64, %25 : !pto.tile_buf<vec, 16x512xf32>, !pto.pipe) {split = 1}
      pto.tmuls ins(%64, %cst : !pto.tile_buf<vec, 16x512xf32>, f32) outs(%64 : !pto.tile_buf<vec, 16x512xf32>)
      pto.trowmax ins(%64, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
      %65 = pto.treshape %37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %66 = pto.treshape %36 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %67 = pto.treshape %41 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %68 = pto.treshape %38 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      %69 = pto.treshape %39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
      pto.tmax ins(%65, %66 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%65 : !pto.tile_buf<vec, 1x16xf32>)
      pto.tsub ins(%66, %65 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%67 : !pto.tile_buf<vec, 1x16xf32>)
      pto.tmuls ins(%65, %cst_2 : !pto.tile_buf<vec, 1x16xf32>, f32) outs(%66 : !pto.tile_buf<vec, 1x16xf32>)
      pto.trowexpandsub ins(%64, %37 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
      pto.texp ins(%67 : !pto.tile_buf<vec, 1x16xf32>) outs(%67 : !pto.tile_buf<vec, 1x16xf32>)
      pto.texp ins(%33 : !pto.tile_buf<vec, 16x512xf32>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
      pto.tmul ins(%68, %67 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%68 : !pto.tile_buf<vec, 1x16xf32>)
      pto.trowsum ins(%33, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
      pto.tadd ins(%68, %69 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%68 : !pto.tile_buf<vec, 1x16xf32>)
      pto.tcvt ins(%33 {rmode = #pto<round_mode CAST_RINT>} : !pto.tile_buf<vec, 16x512xf32>) outs(%34 : !pto.tile_buf<vec, 16x512xf16>)
      pto.tpush_to_aic(%34 : !pto.tile_buf<vec, 16x512xf16>) {id = 30, split = 1}
      pto.tfree(%25 : !pto.pipe) {split = 1}
      %c7 = arith.constant 7 : index
      scf.for %arg3 = %c1 to %c7 step %c1 {
        %c114688_i64_10 = arith.constant 114688 : i64
        %74 = pto.alloc_tile addr = %c114688_i64_10 : !pto.tile_buf<vec, 16x128xf32>
        pto.tpop(%74, %27 : !pto.tile_buf<vec, 16x128xf32>, !pto.pipe) {split = 1}
        pto.trowexpandmul ins(%35, %40 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
        pto.tadd ins(%35, %74 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x128xf32>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
        pto.tfree(%27 : !pto.pipe) {split = 1}
        %c81920_i64_11 = arith.constant 81920 : i64
        %75 = pto.alloc_tile addr = %c81920_i64_11 : !pto.tile_buf<vec, 16x512xf32>
        pto.tpop(%75, %25 : !pto.tile_buf<vec, 16x512xf32>, !pto.pipe) {split = 1}
        pto.tmuls ins(%75, %cst : !pto.tile_buf<vec, 16x512xf32>, f32) outs(%75 : !pto.tile_buf<vec, 16x512xf32>)
        pto.trowmax ins(%75, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
        %76 = pto.treshape %37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        %77 = pto.treshape %36 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        %78 = pto.treshape %40 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        %79 = pto.treshape %38 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        %80 = pto.treshape %39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        pto.tmax ins(%76, %77 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%76 : !pto.tile_buf<vec, 1x16xf32>)
        pto.tsub ins(%77, %76 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%78 : !pto.tile_buf<vec, 1x16xf32>)
        pto.tmuls ins(%76, %cst_2 : !pto.tile_buf<vec, 1x16xf32>, f32) outs(%77 : !pto.tile_buf<vec, 1x16xf32>)
        pto.trowexpandsub ins(%75, %37 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
        pto.texp ins(%78 : !pto.tile_buf<vec, 1x16xf32>) outs(%78 : !pto.tile_buf<vec, 1x16xf32>)
        pto.texp ins(%33 : !pto.tile_buf<vec, 16x512xf32>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
        pto.tmul ins(%79, %78 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%79 : !pto.tile_buf<vec, 1x16xf32>)
        pto.trowsum ins(%33, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
        pto.tadd ins(%79, %80 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%79 : !pto.tile_buf<vec, 1x16xf32>)
        pto.tcvt ins(%33 {rmode = #pto<round_mode CAST_RINT>} : !pto.tile_buf<vec, 16x512xf32>) outs(%34 : !pto.tile_buf<vec, 16x512xf16>)
        pto.tpush_to_aic(%34 : !pto.tile_buf<vec, 16x512xf16>) {id = 30, split = 1}
        pto.tfree(%25 : !pto.pipe) {split = 1}
        %c114688_i64_12 = arith.constant 114688 : i64
        %81 = pto.alloc_tile addr = %c114688_i64_12 : !pto.tile_buf<vec, 16x128xf32>
        pto.tpop(%81, %27 : !pto.tile_buf<vec, 16x128xf32>, !pto.pipe) {split = 1}
        pto.trowexpandmul ins(%35, %41 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
        pto.tadd ins(%35, %81 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x128xf32>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
        pto.tfree(%27 : !pto.pipe) {split = 1}
        %c81920_i64_13 = arith.constant 81920 : i64
        %82 = pto.alloc_tile addr = %c81920_i64_13 : !pto.tile_buf<vec, 16x512xf32>
        pto.tpop(%82, %25 : !pto.tile_buf<vec, 16x512xf32>, !pto.pipe) {split = 1}
        pto.tmuls ins(%82, %cst : !pto.tile_buf<vec, 16x512xf32>, f32) outs(%82 : !pto.tile_buf<vec, 16x512xf32>)
        pto.trowmax ins(%82, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
        %83 = pto.treshape %37 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        %84 = pto.treshape %36 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        %85 = pto.treshape %41 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        %86 = pto.treshape %38 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        %87 = pto.treshape %39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major> -> !pto.tile_buf<vec, 1x16xf32>
        pto.tmax ins(%83, %84 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%83 : !pto.tile_buf<vec, 1x16xf32>)
        pto.tsub ins(%84, %83 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%85 : !pto.tile_buf<vec, 1x16xf32>)
        pto.tmuls ins(%83, %cst_2 : !pto.tile_buf<vec, 1x16xf32>, f32) outs(%84 : !pto.tile_buf<vec, 1x16xf32>)
        pto.trowexpandsub ins(%82, %37 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
        pto.texp ins(%85 : !pto.tile_buf<vec, 1x16xf32>) outs(%85 : !pto.tile_buf<vec, 1x16xf32>)
        pto.texp ins(%33 : !pto.tile_buf<vec, 16x512xf32>) outs(%33 : !pto.tile_buf<vec, 16x512xf32>)
        pto.tmul ins(%86, %85 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%86 : !pto.tile_buf<vec, 1x16xf32>)
        pto.trowsum ins(%33, %32 : !pto.tile_buf<vec, 16x512xf32>, !pto.tile_buf<vec, 16x512xf32>) outs(%39 : !pto.tile_buf<vec, 16x1xf32, blayout=col_major>)
        pto.tadd ins(%86, %87 : !pto.tile_buf<vec, 1x16xf32>, !pto.tile_buf<vec, 1x16xf32>) outs(%86 : !pto.tile_buf<vec, 1x16xf32>)
        pto.tcvt ins(%33 {rmode = #pto<round_mode CAST_RINT>} : !pto.tile_buf<vec, 16x512xf32>) outs(%34 : !pto.tile_buf<vec, 16x512xf16>)
        pto.tpush_to_aic(%34 : !pto.tile_buf<vec, 16x512xf16>) {id = 30, split = 1}
        pto.tfree(%25 : !pto.pipe) {split = 1}
      }
      %c114688_i64_8 = arith.constant 114688 : i64
      %70 = pto.alloc_tile addr = %c114688_i64_8 : !pto.tile_buf<vec, 16x128xf32>
      pto.tpop(%70, %27 : !pto.tile_buf<vec, 16x128xf32>, !pto.pipe) {split = 1}
      pto.trowexpandmul ins(%35, %40 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
      pto.tadd ins(%35, %70 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x128xf32>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
      pto.tfree(%27 : !pto.pipe) {split = 1}
      %c114688_i64_9 = arith.constant 114688 : i64
      %71 = pto.alloc_tile addr = %c114688_i64_9 : !pto.tile_buf<vec, 16x128xf32>
      pto.tpop(%71, %27 : !pto.tile_buf<vec, 16x128xf32>, !pto.pipe) {split = 1}
      pto.trowexpandmul ins(%35, %41 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
      pto.tadd ins(%35, %71 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x128xf32>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
      pto.tfree(%27 : !pto.pipe) {split = 1}
      pto.trowexpanddiv ins(%35, %38 : !pto.tile_buf<vec, 16x128xf32>, !pto.tile_buf<vec, 16x1xf32, blayout=col_major>) outs(%35 : !pto.tile_buf<vec, 16x128xf32>)
      %72 = arith.addi %43, %31 : index
      %73 = pto.partition_view %42, offsets = [%72, %c0], sizes = [%c16, %c128] : !pto.tensor_view<?x?xf32>
      pto.tstore ins(%35 : !pto.tile_buf<vec, 16x128xf32>) outs(%73 : !pto.partition_tensor_view<16x128xf32>)
    }
    return
  }
  func.func @call_both(%arg0: memref<256xi64>, %arg1: !pto.ptr<f32>, %arg2: !pto.ptr<f16>, %arg3: !pto.ptr<f16>, %arg4: !pto.ptr<f16>, %arg5: !pto.ptr<f32>) attributes {pto.entry} {
    pto.set_ffts %arg0 : memref<256xi64>
    call @cube_kernel(%arg1, %arg2, %arg3, %arg4) : (!pto.ptr<f32>, !pto.ptr<f16>, !pto.ptr<f16>, !pto.ptr<f16>) -> ()
    call @vector_kernel(%arg1, %arg5) : (!pto.ptr<f32>, !pto.ptr<f32>) -> ()
    return
  }
}

