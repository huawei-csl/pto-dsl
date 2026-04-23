/**
 * Host launcher for ``sinkhorn_k4_fp16`` emitted from ``sinkhorn_k4_builder.py``.
 * Exports ``call_sinkhorn`` for ctypes / torch_npu.
 */
#ifndef KERNEL_CPP
#define KERNEL_CPP "outputs/sinkhorn_k4_generated.cpp"
#endif
#include KERNEL_CPP

extern "C" void call_sinkhorn(
    uint32_t cube_core_num,
    void *stream,
    uint8_t *input,
    uint8_t *output,
    uint32_t num_matrices,
    uint32_t repeat,
    float eps) {
  sinkhorn_k4_fp16<<<cube_core_num * 2, nullptr, stream>>>(
      (__gm__ half *)input,
      (__gm__ half *)output,
      static_cast<int32_t>(num_matrices),
      static_cast<int32_t>(repeat),
      eps);
}
