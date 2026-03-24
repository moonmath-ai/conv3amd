/**
 * float8_e4m3fn stub: per-block LDS holds three H tubelets (h-1, h, h+1); zero-pad missing neighbors.
 * Loads input at h_idx-1 / h_idx / h_idx+1; output is still the center tubelet only (identity).
 *
 * Grid: (x, y, z) = (H tubelet index, C_out/16 groups, batch). Threads map to W (coalesced).
 */

#include <torch/extension.h>
#include <rocwmma/rocwmma.hpp>

constexpr int kMFMAInputDim = 16;
constexpr int kMFMAOutputDim = 16;
constexpr int kMFMAInternalDim = 32;
constexpr int kNumC = 128;
constexpr int kPatchH = 192;
constexpr int kPatchW = 320;
constexpr int kPatchT = 4;
constexpr int kKernel = 3;
constexpr int kFilter = 27; // kKernel * kKernel * kKernel = 27

constexpr int kBlockSize = kPatchW;
constexpr int kChannelsPerOutGroup = kMFMAOutputDim; // 16
constexpr int kFilterPadded = kMFMAInternalDim; // 32
constexpr int kPatchWPerMFMAExe = kPatchW / kMFMAInputDim; // 20
constexpr int kFiltersLoadedInFirstTxWave = 11; // 11 * kFilter = 297 <= kBlockSize

namespace {
__global__ void conv3d_my_global_lds_kernel(
  const char* __restrict__ input,
  const char* __restrict__ weight,
  char* __restrict__ output) {

  __shared__ char tubelet[kPatchT * 3 * kPatchW];
  __shared__ char samples[kMFMAInputDim][kMFMAInternalDim];  // [16][32] for MFMA B(32x16)
  __shared__ char filters[kChannelsPerOutGroup][kFilterPadded]; // [16][32] for MFMA A(16x32)
  __shared__ char accumulators[kChannelsPerOutGroup][kPatchT * kPatchW];

  const int b_idx = blockIdx.z;
  const int h_idx = blockIdx.x;
  const int outgrp_idx = blockIdx.y;

  const int tx = threadIdx.x;
  const int wave_idx = tx / 64;
  const int wave_tx = tx % 64;
  const int wgrp = tx / 320;
  const int wgrp_tx = tx % 320;

  // // zero pad group filters tail bytes once
  // for (int pid = tx; pid < kChannelsPerOutGroup * (kFilterPadded - kFilter); pid += kBlockSize) {
  //   const int c_idx = pid / (kFilterPadded - kFilter);
  //   const int col = kFilter + (pid % (kFilterPadded - kFilter));
  //   filters[c_idx][col] = 0;
  // }
  // // __syncthreads();

  // loop over all 128 input channels
  for (int cin_idx = 0; cin_idx < kNumC; ++cin_idx) {

    // // load group filters from global
    // // TODO: better coalescing if order is (cin, cout) instead of (cout, cin)
    // #pragma unroll
    // for (int c_idx = 0; c_idx < kFiltersLoadedInFirstTxWave; ++c_idx) {
    //   const int cout_idx = outgrp_idx * kChannelsPerOutGroup + c_idx;
    //   const int filter_off = c_idx * kFilter;
    //   if ((tx >= filter_off) && (tx < filter_off + kFilter)) {
    //     const int offset = (cout_idx * kNumC + cin_idx) * kFilter + (tx - filter_off);
    //     filters[c_idx][tx - filter_off] = weight[offset];
    //   }
    // }
    // #pragma unroll
    // for (int c_idx = kFiltersLoadedInFirstTxWave; c_idx < kChannelsPerOutGroup; ++c_idx) {
    //   const int cout_idx = outgrp_idx * kChannelsPerOutGroup + c_idx;
    //   const int filter_off = (c_idx - kFiltersLoadedInFirstTxWave) * kFilter;
    //   if ((tx >= filter_off) && (tx < filter_off + kFilter)) {
    //     const int offset = (cout_idx * kNumC + cin_idx) * kFilter + (tx - filter_off);
    //     filters[c_idx][tx - filter_off] = weight[offset];
    //   }
    // }
    // // __syncthreads();

    // load tubelet from global
    #pragma unroll
    for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
      const int base = t_idx * 3 * kPatchW + tx;
      // strip h-1
      if (h_idx > 0) {
        const int offset = (((b_idx * kNumC + cin_idx) * kPatchT + t_idx) * kPatchH + (h_idx - 1)) * kPatchW + tx;
        tubelet[base] = input[offset];
      } else {
        tubelet[base] = 0;
      }
      // strip h
      {
        const int offset = (((b_idx * kNumC + cin_idx) * kPatchT + t_idx) * kPatchH + h_idx) * kPatchW + tx;
        tubelet[base + kPatchW] = input[offset];
      }
      // strip h+1
      if (h_idx + 1 < kPatchH) {
        const int offset = (((b_idx * kNumC + cin_idx) * kPatchT + t_idx) * kPatchH + (h_idx + 1)) * kPatchW + tx;
        tubelet[base + 2 * kPatchW] = input[offset];
      } else {
        tubelet[base + 2 * kPatchW] = 0;
      }
    }

    // // convolution
    // // load filters to A matrix (16x32)
    // // A: row-major; A(16,32) <- filters[:16][:32]
    // rocwmma::fragment<rocwmma::matrix_a, kMFMAInputDim, kMFMAOutputDim, kMFMAInternalDim, rocwmma::float8_fnuz_t, rocwmma::row_major> frag_a;
    // rocwmma::load_matrix_sync(frag_a, reinterpret_cast<const rocwmma::float8_fnuz_t*>(&filters[0][0]), static_cast<uint32_t>(kFilterPadded));

    // // process tubelet across all output group channels
    // for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
    //   for (int w_idx = 0; w_idx < kPatchWPerMFMAExe; ++w_idx) {
    //     // load accumulators chunk to C matrix (16x16) - zero on first input channel
    //     // C: row-major; C(16,16) <- accumulators[:16][t*kPatchW+w*kMFMAInputDim:+16]
    //     rocwmma::fragment<rocwmma::accumulator, kMFMAInputDim, kMFMAOutputDim, kMFMAInternalDim, rocwmma::float8_fnuz_t, rocwmma::row_major> frag_c;
    //     if (cin_idx == 0) {
    //       rocwmma::fill_fragment(frag_c, rocwmma::float8_fnuz_t{});
    //     } else {
    //       rocwmma::load_matrix_sync(frag_c, reinterpret_cast<const rocwmma::float8_fnuz_t*>(&accumulators[0][t_idx * kPatchW + w_idx * kMFMAInputDim]), static_cast<uint32_t>(kPatchT * kPatchW));
    //     }

    //     // load tubelet chunk to B matrix (32x16)
    //     // initially move to samples
    //     // for (int s_idx = 0; s_idx < kMFMAInputDim; ++s_idx) {
    //     //   samples[s_idx][w_idx * kMFMAInputDim + tx] = tubelet[t_idx * 3 * kPatchW + kPatchW + tx];
    //     // }
    //     rocwmma::fragment<rocwmma::matrix_b, kMFMAInputDim, kMFMAOutputDim, kMFMAInternalDim, rocwmma::float8_fnuz_t, rocwmma::col_major> frag_b;
    //     rocwmma::load_matrix_sync(frag_b, reinterpret_cast<const rocwmma::float8_fnuz_t*>(&samples[0][0]), static_cast<uint32_t>(kMFMAInternalDim), rocwmma::layout_t::mem_row_major);

    //     // MFMA A(16x32) x B(32x16) + C(16x16) -> D(16x16)
    //     rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    //     // store accumulators chunk from D matrix (16x16) including fp32->fp8 conversion
    //     rocwmma::store_matrix_sync(reinterpret_cast<rocwmma::float8_fnuz_t*>(&accumulators[0][t_idx * kPatchW + w_idx * kMFMAInputDim]), frag_c, static_cast<uint32_t>(kPatchT * kPatchW));
    //   }
    // }

    // temporary accumulators assignment (sum all)
    const int cout_idx = cin_idx % kChannelsPerOutGroup;
    #pragma unroll
    for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
      const int base = t_idx * kPatchW + tx;
      const int offset = t_idx * 3 * kPatchW + tx;
      accumulators[cout_idx][base] += (tubelet[offset] + tubelet[offset + kPatchW] + tubelet[offset + 2 * kPatchW]);
    }
  }

  // loop over 16 output channels
  #pragma unroll
  for (int cout_idx = 0; cout_idx < kChannelsPerOutGroup; ++cout_idx) {
    // store accumulators to global
    #pragma unroll
    for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
      const int base = (((b_idx * kNumC + outgrp_idx * kChannelsPerOutGroup + cout_idx) * kPatchT + t_idx) * kPatchH + h_idx) * kPatchW + tx;
      const int offset = t_idx * kPatchW + tx;
      output[base] = accumulators[cout_idx][offset];
    }
  }

} // conv3d_my_global_lds_kernel
} // namespace


torch::Tensor launch_conv3d_my_kernel(
    torch::Tensor input,
    torch::Tensor weight) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float8_e4m3fn, "conv3d_my_kernel: expected float8_e4m3fn input, got ", input.scalar_type());
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float8_e4m3fn, "conv3d_my_kernel: expected float8_e4m3fn weight, got ", weight.scalar_type());
  TORCH_CHECK(weight.size(1) == input.size(1), "weight in_channels (", weight.size(1), ") must match input in_channels (", input.size(1), ")");

  const int64_t B = input.size(0);
  const int64_t C_in = input.size(1);
  const int64_t T_in = input.size(2);
  const int64_t H_in = input.size(3);
  const int64_t W_in = input.size(4);
  const int64_t C_out = weight.size(0);

  TORCH_CHECK(B >= 1, "conv3d_my_kernel: batch B must be >= 1");
  TORCH_CHECK(C_in == kNumC && C_out == kNumC, "conv3d_my_kernel: C_in and C_out must be ", kNumC, " (identity tubelet stub)");
  TORCH_CHECK(T_in == kPatchT && H_in == kPatchH && W_in == kPatchW, "conv3d_my_kernel: input (T,H,W) must be (", kPatchT, ",", kPatchH, ",", kPatchW, ")");
  TORCH_CHECK(weight.size(2) == kKernel && weight.size(3) == kKernel && weight.size(4) == kKernel, "conv3d_my_kernel: weight kernel must be ", kKernel, "x", kKernel, "x", kKernel);

  TORCH_CHECK(C_out % kChannelsPerOutGroup == 0, "conv3d_my_kernel: C_out (", C_out, ") must be divisible by ", kChannelsPerOutGroup);

  const int64_t C_out_per_group = C_out / kChannelsPerOutGroup;
  const int64_t T_out = T_in;
  const int64_t H_out = H_in;
  const int64_t W_out = W_in;

  const unsigned int grid_x = H_out;
  const unsigned int grid_y = C_out_per_group;
  const unsigned int grid_z = B;

  auto out_options = input.options().dtype(input.dtype());
  auto output = torch::empty({B, C_out, T_out, H_out, W_out}, out_options);

  dim3 grid(grid_x, grid_y, grid_z);
  conv3d_my_global_lds_kernel<<<grid, kBlockSize>>>(
      reinterpret_cast<const char*>(input.data_ptr()),
      reinterpret_cast<const char*>(weight.data_ptr()),
      reinterpret_cast<char*>(output.data_ptr()));

  return output;
}
