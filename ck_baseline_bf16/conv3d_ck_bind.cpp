#include <torch/extension.h>

#include "conv3d_ck_launch.h"

torch::Tensor conv3d_bf16_ck_forward_py(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t pad_left_d,
    int64_t pad_left_h,
    int64_t pad_left_w,
    int64_t pad_right_d,
    int64_t pad_right_h,
    int64_t pad_right_w) {
  return conv3d_bf16_ck_forward(
      input,
      weight,
      pad_left_d,
      pad_left_h,
      pad_left_w,
      pad_right_d,
      pad_right_h,
      pad_right_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "conv3d_bf16_ck_forward",
      &conv3d_bf16_ck_forward_py,
      "BF16 grouped conv3d via CK (explicit v3 instances).",
      py::arg("input"),
      py::arg("weight"),
      py::arg("pad_left_d"),
      py::arg("pad_left_h"),
      py::arg("pad_left_w"),
      py::arg("pad_right_d"),
      py::arg("pad_right_h"),
      py::arg("pad_right_w"));
}
