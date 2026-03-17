/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
All rights reserved.

See LICENSE in the root of the software repository:
https://github.com/huawei-csl/pto-kernels/
for the full License text.
*/
#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

#include "aclrtlaunch_tri_inv_trick_fp16.h"
#include "utils.h"

namespace pto_isa_ops {

/**
 * @brief Triangular inverse using the "inverse trick" method.
 *
 * @param [in] M Input tensor containing square matrices on the last two
 * dimensions.
 * @return at::Tensor Tensor containing inverses of input matrices.
 */
at::Tensor run_tri_inv_trick(const at::Tensor& M) {
  const at::Device device = M.options().device();
  const auto dtype = M.options().dtype();
  const auto dtype_out = at::kFloat;
  const uint32_t max_block_size = 16;
  if (!(dtype == at::kHalf)) {
    throw std::runtime_error(
        "Unsupported dtype for tri_inv_trick kernel. Supports only fp16");
  }

  const uint32_t matrix_size = static_cast<uint32_t>(M.size(-1));
  if (matrix_size != M.size(-2)) {
    throw std::runtime_error("Only square matrices are supported.\n");
  }
  const uint32_t num_elems = static_cast<uint32_t>(M.numel());
  const uint32_t block_dim =
      static_cast<uint32_t>(num_elems / (matrix_size * matrix_size));

  const at::Tensor M_inv =
      at::zeros_like(M, at::TensorOptions().dtype(dtype_out).device(device));

  const at::Tensor I_neg =
      at::zeros({matrix_size, matrix_size},
                at::TensorOptions().dtype(dtype).device(device));
  I_neg.fill_diagonal_(-1);
  if (dtype == at::kHalf) {
    EXEC_KERNEL_CMD(tri_inv_trick_fp16, block_dim, M_inv, M, I_neg, matrix_size,
                    max_block_size);
  }

  return M_inv;
}
}  // namespace pto_isa_ops
