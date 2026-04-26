# fp4_gemm — PTO DSL port

Same algorithm and design as `fp8_gemm` (see [../fp8_gemm/README.md](../fp8_gemm/README.md))
but with `BLOCK_K = 32` (= the FP4 weight-group size).

Per-block scale fusion is performed host-side via FP16 pre-scale; the
on-device kernel is an FP16 GEMM with FP32 accumulator. Tests use
non-trivial random `Sa, Sb` to exercise the full fusion path. 4/4 pass.
