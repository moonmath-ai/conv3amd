/**
 * float8_e4m3fn stub: per-block LDS holds three H tubelets (h-1, h, h+1); zero-pad missing neighbors.
 * Loads input at h_ctr_idx-1 / h_ctr_idx / h_ctr_idx+1; output is still the center tubelet only (identity).
 *
 * Grid: (x, y, z) = (H tubelet index, C_out/16 groups, batch). Threads map to W (coalesced).
 */

#include <torch/extension.h>
#include <rocwmma/rocwmma.hpp>

constexpr int kMFMAInputDim = 16;
constexpr int kMFMAOutputDim = 16;
constexpr int kMFMAInternalDim = 32;

constexpr int kNumC = 128;
constexpr int kPatchT = 4;
constexpr int kPatchH = 192;
constexpr int kPatchW = 320;
constexpr int kPatchWDiv4 = kPatchW / 4; // 80
constexpr int kPatchWDiv16 = kPatchW / 16; // 20
constexpr int kKernel = 3;
constexpr int kFilter = 27; // kKernel * kKernel * kKernel = 27

constexpr int kBlockSize = kPatchW;
constexpr int kChannelsPerOutGroup = kMFMAOutputDim; // 16

// constexpr int kFilterPadded = kMFMAInternalDim; // 32
// constexpr int kPatchWPerMFMAExe = kPatchW / kMFMAInputDim; // 20
// constexpr int kFiltersLoadedInFirstTxWave = 11; // 11 * kFilter = 297 <= kBlockSize

namespace {

__device__ inline void async_global_to_lds_4b(
  char __attribute__((address_space(3)))* smem_dst,
  const char __attribute__((address_space(1)))* global_src) {
  __builtin_amdgcn_global_load_lds(
      (const void __attribute__((address_space(1)))*)global_src,
      (void __attribute__((address_space(3)))*)smem_dst,
      4u,
      0u,
      0u);
}

__global__ void conv3d_my_global_lds_kernel(
  const char* __restrict__ input,
  const char* __restrict__ weight,
  char* __restrict__ output) {

  const int b_idx = blockIdx.z; // batch index
  const int h_ctr_idx = blockIdx.x; // height index
  const int outgrp_idx = blockIdx.y; // output group index
  const int tx = threadIdx.x; // thread index
  const int wave_idx = tx / 64; // wave index
  const int wave_tx = tx % 64; // wave thread index

  // __shared__ char samples[kMFMAInputDim][kMFMAInternalDim];  // [16][32] for MFMA B(32x16)
  // __shared__ char filters[kChannelsPerOutGroup][kFilterPadded]; // [16][32] for MFMA A(16x32)
  // // zero pad group filters tail bytes once
  // for (int pid = tx; pid < kChannelsPerOutGroup * (kFilterPadded - kFilter); pid += kBlockSize) {
  //   const int c_idx = pid / (kFilterPadded - kFilter);
  //   const int col = kFilter + (pid % (kFilterPadded - kFilter));
  //   filters[c_idx][col] = 0;
  // }

  __shared__ char tubelet[2][kPatchT * kKernel * kPatchW] __attribute__((aligned(16))); // tubelet double buffer
  __shared__ char accumulators[kChannelsPerOutGroup][kPatchT * kPatchW] __attribute__((aligned(16))); // output accumulators
  #pragma unroll
  for (int i = tx; i < kChannelsPerOutGroup * kPatchT * kPatchWDiv16; i += kBlockSize) {
    reinterpret_cast<int4*>(&accumulators[0][0])[i] = make_int4(0, 0, 0, 0);
  }
  __syncthreads();

  //////////////////////////////////////////////////////////////
  // prolog: produce first
  int cin_idx = 0;
  int buf = 0;
  #pragma unroll
  for (int i = tx; i < kPatchT * kKernel * kPatchWDiv4; i += kBlockSize) {
    const int w_chunk = i % kPatchWDiv4;
    const int k_idx   = (i / kPatchWDiv4) % kKernel;
    const int h_idx   = h_ctr_idx + k_idx - 1;
    const int t_idx   = (i / kPatchWDiv4) / kKernel;

    const int tptr = (t_idx * kKernel + k_idx) * kPatchW + w_chunk * 4;
    if (h_idx < 0 || h_idx >= kPatchH) {
      // zero pad when h_idx is oob
      *reinterpret_cast<int*>(&tubelet[buf][tptr]) = 0;
    } else {
      // load from global to lds
      const size_t iptr = ((static_cast<size_t>(b_idx * kNumC + cin_idx) * kPatchT + t_idx) * kPatchH + static_cast<size_t>(h_idx)) * kPatchW + static_cast<size_t>(w_chunk * 4);
      async_global_to_lds_4b(
          (char __attribute__((address_space(3)))*)&tubelet[buf][tptr],
          (const char __attribute__((address_space(1)))*)(input + iptr));
    }
  }

  //////////////////////////////////////////////////////////////
  // main loop: produce next + consume current
  while (cin_idx < kNumC - 1) {
    const int curr_cin_idx = cin_idx;
    const int curr_buf = buf;

    // sync next buf
    asm volatile("s_barrier" ::: "memory");

    // produce next buf
    cin_idx++;
    buf = (buf + 1) % 2;
    #pragma unroll
    for (int i = tx; i < kPatchT * kKernel * kPatchWDiv4; i += kBlockSize) {
      const int w_chunk = i % kPatchWDiv4;
      const int k_idx   = (i / kPatchWDiv4) % kKernel;
      const int h_idx   = h_ctr_idx + k_idx - 1;
      const int t_idx   = (i / kPatchWDiv4) / kKernel;
  
      const int tptr = (t_idx * kKernel + k_idx) * kPatchW + w_chunk * 4;
      if (h_idx < 0 || h_idx >= kPatchH) {
        // zero pad when h_idx is oob
        *reinterpret_cast<int*>(&tubelet[buf][tptr]) = 0;
      } else {
        // load from global to lds
        const size_t iptr = ((static_cast<size_t>(b_idx * kNumC + cin_idx) * kPatchT + t_idx) * kPatchH + static_cast<size_t>(h_idx)) * kPatchW + static_cast<size_t>(w_chunk * 4);
        async_global_to_lds_4b(
          (char __attribute__((address_space(3)))*)&tubelet[buf][tptr],
          (const char __attribute__((address_space(1)))*)(input + iptr));
      }
    }

    // sync current buf
    asm volatile("s_waitcnt vmcnt(3) lgkmcnt(3)");
    asm volatile("s_barrier" ::: "memory");
  
    // consume current buf
    const int cout_idx = curr_cin_idx % kChannelsPerOutGroup;
    #pragma unroll
    for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
      const int aptr = t_idx * kPatchW + tx;
      const int tptr = t_idx * kKernel * kPatchW + tx;
      accumulators[cout_idx][aptr] += (tubelet[curr_buf][tptr] + tubelet[curr_buf][tptr + kPatchW] + tubelet[curr_buf][tptr + 2 * kPatchW]);
    }
  } // cin_idx loop

  //////////////////////////////////////////////////////////////
  // epilog: consume last
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
  asm volatile("s_barrier" ::: "memory");
  const int cout_idx = cin_idx % kChannelsPerOutGroup;
  #pragma unroll
  for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
    const int aptr = t_idx * kPatchW + tx;
    const int tptr = t_idx * kKernel * kPatchW + tx;
    accumulators[cout_idx][aptr] += (tubelet[buf][tptr] + tubelet[buf][tptr + kPatchW] + tubelet[buf][tptr + 2 * kPatchW]);
  }

  //////////////////////////////////////////////////////////////
  // output loop: 16 output channels
  #pragma unroll
  for (int cout_idx = 0; cout_idx < kChannelsPerOutGroup; ++cout_idx) {
    // store accumulators to global
    #pragma unroll
    for (int t_idx = 0; t_idx < kPatchT; ++t_idx) {
      const int base = (((b_idx * kNumC + outgrp_idx * kChannelsPerOutGroup + cout_idx) * kPatchT + t_idx) * kPatchH + h_ctr_idx) * kPatchW + tx;
      const int offset = t_idx * kPatchW + tx;
      output[base] = accumulators[cout_idx][offset];
    }
  } // cout_idx loop

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
