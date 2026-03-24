/**
 * CausalConv3d "my" implementation: Python-callable 3D conv, float8_e4m3fn only.
 * No padding in this API; padding is done implicitly by the caller/kernel.
 */

#include <torch/extension.h>
#include "conv3d_my_kernel.h"

torch::Tensor conv3d_fp8_forward(torch::Tensor input, torch::Tensor weight) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(input.dtype() == torch::kFloat8_e4m3fn, "input must be float8_e4m3fn");
  TORCH_CHECK(weight.dtype() == torch::kFloat8_e4m3fn, "weight must be float8_e4m3fn");
  TORCH_CHECK(input.dim() == 5 && weight.dim() == 5, "expected 5D input and weight");

  return launch_conv3d_my_kernel(input, weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "conv3d_fp8_forward",
      &conv3d_fp8_forward,
      "Conv3d float8_e4m3fn forward (input, weight).",
      py::arg("input"),
      py::arg("weight"));
}
