import torch
import torch_npu
from ptodsl import jit, pto, tile
from ptodsl import scalar as s
from ptodsl.npu_info import get_test_device

TILE_SIZE = 64

const = s.const


def meta_data():
    dtype = pto.float32
    ptr_type = pto.PtrType(dtype)
    ffts_type = pto.ffts_type
    len_type = pto.int32

    tensor_type = pto.TensorType(rank=2, dtype=dtype)

    subtensor_type_u = pto.SubTensorType(shape=[TILE_SIZE, TILE_SIZE], dtype=dtype)
    subtensor_type_a = pto.SubTensorType(shape=[TILE_SIZE, TILE_SIZE], dtype=dtype)
    subtensor_type_c = pto.SubTensorType(shape=[TILE_SIZE, TILE_SIZE], dtype=dtype)
    subtensor_type_row = pto.SubTensorType(shape=[1, TILE_SIZE], dtype=dtype)

    tile_cfg_mat = pto.TileBufConfig(blayout="ColMajor", slayout="RowMajor")
    tile_cfg_left = pto.TileBufConfig(blayout="RowMajor", slayout="RowMajor")
    tile_cfg_right = pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor")
    tile_cfg_acc = pto.TileBufConfig(blayout="ColMajor", slayout="RowMajor")

    tile_type_a_l1 = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="MAT",
        config=tile_cfg_mat,
    )
    tile_type_u_l1 = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="MAT",
        config=tile_cfg_mat,
    )

    tile_type_a = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="LEFT",
        config=tile_cfg_left,
    )
    tile_type_u = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="RIGHT",
        config=tile_cfg_right,
    )
    tile_type_c = pto.TileBufType(
        shape=[TILE_SIZE, TILE_SIZE],
        dtype=dtype,
        memory_space="ACC",
        config=tile_cfg_acc,
    )

    tile_type_row = pto.TileBufType(
        shape=[1, TILE_SIZE],
        valid_shape=[1, TILE_SIZE],
        dtype=dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )
    tile_type_1x8 = pto.TileBufType(
        shape=[1, 8],
        valid_shape=[1, 8],
        dtype=dtype,
        memory_space="VEC",
        config=pto.TileBufConfig(),
    )

    return {
        "ptr_type": ptr_type,
        "ffts_type": ffts_type,
        "len_type": len_type,
        "tensor_type": tensor_type,
        "subtensor_type_u": subtensor_type_u,
        "subtensor_type_a": subtensor_type_a,
        "subtensor_type_c": subtensor_type_c,
        "subtensor_type_row": subtensor_type_row,
        "tile_type_a_l1": tile_type_a_l1,
        "tile_type_u_l1": tile_type_u_l1,
        "tile_type_a": tile_type_a,
        "tile_type_u": tile_type_u,
        "tile_type_c": tile_type_c,
        "tile_type_row": tile_type_row,
        "tile_type_1x8": tile_type_1x8,
    }


@jit(meta_data=meta_data, block_dim=1, enable_insert_sync=False, init_ffts="ffts_addr")
def run_scan_kernel(
    x_ptr: "ptr_type",
    y_ptr: "ptr_type",
    u_ptr: "ptr_type",
    total_len_i32: "len_type",
) -> None:
    c0 = const(0)
    c1 = const(1)
    cLAST_ROW_ELEM = const(TILE_SIZE - 1)
    cTILE_SIZE = const(TILE_SIZE)
    cN_TILE_ELEM = const(TILE_SIZE * TILE_SIZE)
    c0f = const(0.0, s.float32)
    c0i64 = const(0, s.int64)

    total_len = s.index_cast(total_len_i32)
    num_tiles = total_len // cN_TILE_ELEM

    with pto.cube_section():
        tvX_cube = pto.as_tensor(
            tensor_type,
            ptr=x_ptr,
            shape=[cTILE_SIZE, cTILE_SIZE],
            strides=[cTILE_SIZE, cTILE_SIZE],
            layout="ND",
        )
        tvU = pto.as_tensor(
            tensor_type,
            ptr=u_ptr,
            shape=[cTILE_SIZE, cTILE_SIZE],
            strides=[cTILE_SIZE, cTILE_SIZE],
            layout="ND",
        )
        tvOut_cube = pto.as_tensor(
            tensor_type,
            ptr=y_ptr,
            shape=[cTILE_SIZE, cTILE_SIZE],
            strides=[cTILE_SIZE, cTILE_SIZE],
            layout="ND",
        )

        uTileL1 = pto.alloc_tile(tile_type_u_l1)
        uTile = pto.alloc_tile(tile_type_u)
        aTileL1 = pto.alloc_tile(tile_type_a_l1)
        aTile = pto.alloc_tile(tile_type_a)
        cTile = pto.alloc_tile(tile_type_c)

        svU = pto.slice_view(
            subtensor_type_u,
            source=tvU,
            offsets=[c0, c0],
            sizes=[cTILE_SIZE, cTILE_SIZE],
        )
        pto.load(svU, uTileL1)
        pto.record_wait_pair("LOAD", "MOV_M2L", 0)

        tile.mov(uTileL1, uTile)
        pto.record_wait_pair("MOV_M2L", "MATMUL", 0)

        for tile_idx in pto.range(c0, num_tiles, c1):
            offset = tile_idx * cTILE_SIZE
            svX = pto.slice_view(
                subtensor_type_a,
                source=tvX_cube,
                offsets=[offset, c0],
                sizes=[cTILE_SIZE, cTILE_SIZE],
            )
            svOut = pto.slice_view(
                subtensor_type_c,
                source=tvOut_cube,
                offsets=[offset, c0],
                sizes=[cTILE_SIZE, cTILE_SIZE],
            )

            pto.load(svX, aTileL1)
            pto.record_wait_pair("LOAD", "MOV_M2L", 1)

            tile.mov(aTileL1, aTile)
            pto.record_wait_pair("MOV_M2L", "MATMUL", 1)

            tile.matmul(aTile, uTile, cTile)
            pto.record_wait_pair("MATMUL", "STORE_ACC", 1)

            pto.store(cTile, svOut)
            pto.record_wait_pair("STORE_ACC", "LOAD", 2)

            pto.sync_set(pto.PIPE_FIX, 0)
            pto.sync_wait(pto.PIPE_MTE3, 1)

    with pto.vector_section():
        tvOut_vec = pto.as_tensor(
            tensor_type,
            ptr=y_ptr,
            shape=[total_len // cTILE_SIZE, cTILE_SIZE],
            strides=[cTILE_SIZE, c1],
        )

        rowTile = pto.alloc_tile(tile_type_row)
        sumTile1x8 = pto.alloc_tile(tile_type_1x8)
        tile.setval(sumTile1x8, c0, c0f)

        for tile_idx in pto.range(c0, num_tiles, c1):
            pto.sync_wait(pto.PIPE_FIX, 0)

            vid = pto.get_subblock_idx()

            with pto.if_context(vid == c0i64):
                tile_offset = tile_idx * cTILE_SIZE
                for r in pto.range(c0, cTILE_SIZE, c1):
                    offset = tile_offset + r
                    svRow = pto.slice_view(
                        subtensor_type_row,
                        source=tvOut_vec,
                        offsets=[offset, c0],
                        sizes=[c1, cTILE_SIZE],
                    )

                    pto.load(svRow, rowTile)

                    pto.record_wait_pair("LOAD", "VEC", 2)

                    # Extract the stateful running_sum from our memory buffer
                    running_sum = tile.getval(sumTile1x8, c0, dtype=s.float32)

                    tile.adds(rowTile, running_sum, rowTile)
                    # Ideally we would synchronize PIPE_S and PIPE_V here, but that is not currently possible
                    # with pto.record_wait_pair, instead we use a barrier
                    # pto.record_wait_pair("PIPE_V", "PIPE_S", 2)
                    pto.barrier(pto.PIPE_ALL)
                    running_sum_next = tile.getval(
                        rowTile, cLAST_ROW_ELEM, dtype=s.float32
                    )
                    # Persist the new running sum back to the memory buffer to loop-carry
                    tile.setval(sumTile1x8, c0, running_sum_next)
                    pto.record_wait_pair("VEC", "STORE_VEC", 2)

                    pto.store(rowTile, svRow)

                    pto.record_wait_pair("STORE_VEC", "LOAD", 3)

                pto.record_wait_pair("LOAD", "VEC", 3)

            pto.sync_set(pto.PIPE_MTE3, 1)


def test_scan(n_tiles=64):
    device = get_test_device()
    torch.npu.set_device(device)

    total_len = TILE_SIZE * TILE_SIZE * n_tiles
    torch.manual_seed(0)
    dtype = torch.float32

    # Prepare Inputs
    x = torch.rand(size=(total_len,), device=device, dtype=dtype).contiguous()
    y = torch.zeros_like(x)

    # Generate upper triangular matrix of 1s (s x s)
    u = torch.triu(
        torch.ones((TILE_SIZE, TILE_SIZE), device=device, dtype=dtype)
    ).contiguous()

    # Expected PyTorch computation
    expected_scan = torch.cumsum(x.cpu(), dim=0)

    # NPU JIT Kernel execution
    repeat_runs = 20
    print(
        f"Running scan for {total_len} elements ({n_tiles} {TILE_SIZE}x{TILE_SIZE} tiles)"
    )
    actual_scan = []
    for _ in range(repeat_runs):
        y.zero_()
        run_scan_kernel(x, y, u, total_len)
        actual_scan.append(y.cpu().clone())

    torch.npu.synchronize()

    # Check for consistency across runs and correctness against the expected count
    repeat_results = []
    for i, scan in enumerate(actual_scan):
        are_close = torch.allclose(scan, expected_scan, rtol=1e-3, atol=1e-3)
        if not are_close:
            unequal_count = torch.sum(scan != expected_scan)
        else:
            unequal_count = 0
        repeat_results.append([are_close, unequal_count])

    has_mismatch = any(not eq for eq, _ in repeat_results)
    if has_mismatch:
        print("Expected:\n", expected_scan[-10:])
        for i, result in enumerate(repeat_results):
            eq, count = result
            if not eq:
                print(
                    f"Inconsistent results run {i}, different elements: {count}/{total_len}. Sample:"
                )
                print(actual_scan[i][-10:])
        raise AssertionError(
            f"Scan mismatch for tile_size={tile_size}, total_len={total_len} ({n_tiles} tiles)"
        )

    print("All results matched. Scan test passed successfully.\n")


if __name__ == "__main__":
    test_scan(1)
    test_scan(16)
    test_scan(64)
    test_scan(100)
    test_scan(1000)
