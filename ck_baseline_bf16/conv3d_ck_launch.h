#pragma once

#include <torch/extension.h>

/**
 * Grouped 3D conv forward (BF16) via CK DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3.
 *
 * input:  (N, C, D, H, W) contiguous bfloat16 — logical NGCDHW with G=1 (matches NCDHW storage)
 * weight: (K, C, kD, kH, kW) contiguous bfloat16 — logical GKCZYX with G=1
 *
 * Per-dimension padding (depth, height, width) for conv; use (0,0,0)x2 if input is pre-padded.
 */
torch::Tensor conv3d_bf16_ck_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    int64_t pad_left_d,
    int64_t pad_left_h,
    int64_t pad_left_w,
    int64_t pad_right_d,
    int64_t pad_right_h,
    int64_t pad_right_w);
