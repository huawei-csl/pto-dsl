from ptodsl import Constexpr, const_expr, pto, range_constexpr, to_ir_module
from ptodsl import scalar as s


const = s.const


def meta_data(TILE_K, UNROLL=2):
    dtype = pto.float32
    return {
        "index_dtype": pto.int32,
        "tile_type": pto.TileBufType(
            shape=[1, TILE_K // 2],
            valid_shape=[1, TILE_K // 2],
            dtype=dtype,
            memory_space="VEC",
        ),
    }


@to_ir_module(meta_data=meta_data)
def constexpr_tile_kernel(
    n: "index_dtype",
    TILE_K: Constexpr[int],
    UNROLL: Constexpr[int] = 2,
) -> None:
    with pto.vector_section():
        if const_expr(TILE_K % 128 == 0):
            for _ in range_constexpr(UNROLL):
                pto.alloc_tile(tile_type)
        else:
            pto.alloc_tile(tile_type)


if __name__ == "__main__":
    print(constexpr_tile_kernel(TILE_K=128, UNROLL=3))
