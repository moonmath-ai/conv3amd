#pragma once

#include <torch/extension.h>

/**
 * Launches the "my" kernel. Input must be float8_e4m3fn; used as-is (no padding in this API).
 */
torch::Tensor launch_conv3d_my_kernel(
    torch::Tensor input,
    torch::Tensor weight);
