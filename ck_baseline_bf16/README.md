# ck_baseline_bf16

BF16 stage-0 causal 3D conv using **Composable Kernel** `DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3` (NGCDHW / GKCZYX / NGKDHW), with a small set of explicit tile configs from CK’s mem-friendly BF16 tables (Intrawave/Interwave, pipeline v2).

## What this launches

- CK’s grouped conv forward v3 path, including **CK’s own** NGCDHW transpose + `kernel_grouped_conv_fwd_xdl_cshuffle_v3`.
- A **rocprof** capture of `torch.nn.Conv3d` may also show **MIOpen** `batched_transpose_32x32_half` / `batched_transpose_32x16_half` — those come from MIOpen’s module, not from this extension. Numerically, compare against `F.conv3d` / `CausalConv3d` in BF16.

## Build

```bash
export CK_ROOT=/path/to/composablekernel   # .../rocm-libraries/projects/composablekernel
cd ck_baseline_bf16
python setup.py build_ext --inplace
```

Requires PyTorch **ROCm**, `hipcc`, and CK headers at `$CK_ROOT/include` and `$CK_ROOT/library/include`. Descriptor lengths/strides are computed in the launcher (no `host_tensor.cpp` link — that TU does not build cleanly with PyTorch’s HIP extension flags).

Compile can be **memory-heavy** (large device templates). If the linker/compiler OOMs, reduce parallelism: `MAX_JOBS=1 python setup.py build_ext --inplace`.

## Test vs PyTorch

From `conv3amd/`:

```bash
python conv3amd.py --test-ck-bf16
```

Or inside this directory:

```bash
python conv3d_baseline_bf16.py
```

## Run driver with CK path

```bash
python conv3amd.py --use-ck-bf16
```
