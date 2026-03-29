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

  // __shared__ char samples[kMFMAInputDim][kMFMAInternalDim];  // [16][32] for MFMA B(32x16)
  // __shared__ char filters[kChannelsPerOutGroup][kFilterPadded]; // [16][32] for MFMA A(16x32)
  __shared__ char tubelet[2][kPatchT * 3 * kPatchW]; // tubelet double buffer
  __shared__ char accumulators[kChannelsPerOutGroup][kPatchT * kPatchW];
  __shared__ int buf_semaphore[2]; // buffer semaphore
  if (threadIdx.x == 0) {
    buf_semaphore[0] = 0; // buffer 0 is empty
    buf_semaphore[1] = 0; // buffer 1 is empty
  }
  __syncthreads(); // wait for all threads to initialize the semaphore
  
  const int b_idx = blockIdx.z; // batch index
  const int h_idx = blockIdx.x; // height index
  const int outgrp_idx = blockIdx.y; // output group index
  const int tx = threadIdx.x; // thread index
  const int wave_idx = tx / 64; // wave index
  const int wave_tx = tx % 64; // wave thread index
  const int wgrp = tx / 320; // wave group index
  const int wgrp_tx = tx % 320; // wave group thread index

  // // zero pad group filters tail bytes once
  // for (int pid = tx; pid < kChannelsPerOutGroup * (kFilterPadded - kFilter); pid += kBlockSize) {
  //   const int c_idx = pid / (kFilterPadded - kFilter);
  //   const int col = kFilter + (pid % (kFilterPadded - kFilter));
  //   filters[c_idx][col] = 0;
  // }

  if (wave_idx == 0) { // producer

    // loop over all 128 input channels
    for (int cin_idx = 0; cin_idx < kNumC; ++cin_idx) {
      const int buf = cin_idx % 2;
      // load tubelet from global

      // acquire buffer semaphore
      if (wave_tx == 0) {
        while (__atomic_load_n(&buf_semaphore[buf], __ATOMIC_ACQUIRE) > 0) { __builtin_amdgcn_s_sleep(1); }
      }
      __builtin_amdgcn_wave_barrier();

      // low edge
      if (h_idx == 0) {
        #pragma unroll
        for (int w_idx = 0; w_idx < 5; ++w_idx) {
          #pragma unroll
          for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
            const int base = t_idx * 3 * kPatchW + w_idx * 64 + tx;
            const int offset = (((b_idx * kNumC + cin_idx) * kPatchT + t_idx) * kPatchH + h_idx) * kPatchW + w_idx * 64 + tx;
            tubelet[buf][base] = 0;
            tubelet[buf][base + kPatchW] = input[offset];
            tubelet[buf][base + 2 * kPatchW] = input[offset + kPatchW];
          }
        }
      } 
      // high edge
      else if (h_idx == kPatchH - 1) {
        #pragma unroll
        for (int w_idx = 0; w_idx < 5; ++w_idx) {
          #pragma unroll
          for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
            const int base = t_idx * 3 * kPatchW + w_idx * 64 + tx;
            const int offset = (((b_idx * kNumC + cin_idx) * kPatchT + t_idx) * kPatchH + h_idx) * kPatchW + w_idx * 64 + tx;
            tubelet[buf][base] = input[offset - kPatchW];
            tubelet[buf][base + kPatchW] = input[offset];
            tubelet[buf][base + 2 * kPatchW] = 0;
          }
        }  
      } 
      // middle
      else {
        #pragma unroll
        for (int w_idx = 0; w_idx < 5; ++w_idx) {
          #pragma unroll
          for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
            const int base = t_idx * 3 * kPatchW + w_idx * 64 + tx;
            const int offset = (((b_idx * kNumC + cin_idx) * kPatchT + t_idx) * kPatchH + h_idx) * kPatchW + w_idx * 64 + tx;
            tubelet[buf][base] = input[offset - kPatchW];
            tubelet[buf][base + kPatchW] = input[offset];
            tubelet[buf][base + 2 * kPatchW] = input[offset + kPatchW];
          }
        }  
      }

      // release buffer semaphore
      if (wave_tx == 0) {
        __atomic_store_n(&buf_semaphore[buf], 4, __ATOMIC_RELEASE);
      }
    } // cin_idx

  } // producer
  else 
  { // consumer x4

    // loop over all 128 input channels
    for (int cin_idx = 0; cin_idx < kNumC; ++cin_idx) {
      const int buf = cin_idx % 2;

      // acquire buffer semaphore
      if (wave_tx == 0) {
        while (__atomic_load_n(&buf_semaphore[buf], __ATOMIC_ACQUIRE) == 0) { __builtin_amdgcn_s_sleep(1); }
      }
      __builtin_amdgcn_wave_barrier();

      // temporary accumulators assignment (sum all)
      const int cout_idx = cin_idx % kChannelsPerOutGroup;
      #pragma unroll
      for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
        const int base = t_idx * kPatchW + tx;
        const int offset = t_idx * 3 * kPatchW + tx;
        accumulators[cout_idx][base] += (tubelet[buf][offset] + tubelet[buf][offset + kPatchW] + tubelet[buf][offset + 2 * kPatchW]);
      }
      const int base = (wave_idx - 1) * kPatchW + wave_tx;
      const int offset = (wave_idx - 1) * 3 * kPatchW + wave_tx;
      accumulators[cout_idx][base] += (tubelet[buf][offset] + tubelet[buf][offset + kPatchW] + tubelet[buf][offset + 2 * kPatchW]);

      // release buffer semaphore
      if (wave_tx == 0) {
        __atomic_fetch_add(&buf_semaphore[buf], -1, __ATOMIC_RELEASE);
      }
    } // cin_idx

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
      const int base = (((b_idx * kNumC + outgrp_idx * kChannelsPerOutGroup + cout_idx) * kPatchT + wave_idx - 1) * kPatchH + h_idx) * kPatchW + wave_tx;
      const int offset = (wave_idx - 1) * kPatchW + wave_tx;
      output[base] = accumulators[cout_idx][offset];
    } // cout_idx

  } // consumer

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
