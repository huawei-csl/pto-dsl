# sparse_attn — PTO DSL port (SKELETON)

Original (TileLang/GPU): top-k indexed FlashAttention with attn_sink bias.
For each (batch, query-position) the kernel:
1. Reads `topk` precomputed KV indices.
2. Gathers KV rows (with `-1` sentinel masked to 0).
3. Streaming softmax (running `m`, `l`, online rescale of `acc_o`).
4. Adds `exp(attn_sink - m)` to `l` before final divide.
5. Writes BF16 output.

## NPU port status: SKELETON

This file lays out the per-(b, m) loop, allocates the running-stats tiles,
and includes detailed `TODO`s where the algorithmic body must be filled in.
The two non-trivial NPU porting concerns are:

1. **Index-gather of KV rows.** GPU does `if_then_else` per-element; on
   Ascend, the cleanest approach is a SCALAR-pipe DMA-by-index pre-pass
   that builds a contiguous `[BLOCK, D]` KV tile in VEC, plus a fp32
   mask tile that is `-inf` on lanes whose index == -1.
2. **q @ k^T inside the streaming loop.** With `H_PAD=16, D=128, BLOCK=64`
   the partial GEMM is small enough to do on the cube (LEFT/RIGHT/ACC tiles)
   per chunk, but a fully VEC implementation via `row_expand_mul` +
   `row_sum` is also viable for small H.

For a complete software-pipelined NPU FlashAttention reference, see
[examples/aot/flash_attention/kernels/fa_builder.py](../../flash_attention/kernels/fa_builder.py).

## Dtype mapping

| GPU         | NPU port     |
|-------------|--------------|
| BF16 q/kv/o | FP16 q/kv/o  |
| FP32 stats  | FP32 (same)  |
| INT32 idx   | INT32 (same) |
