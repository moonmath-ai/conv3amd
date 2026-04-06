# FP8 (`float8_e4m3fn`) CausalConv3d baseline (Python-callable)

This directory provides a **Python-callable** causal 3D conv in **`torch.float8_e4m3fn` only** (no BF16 path). Replace the stub kernel with your HIP implementation when ready.

## Layout

- **`conv3d_my.cpp`** — C++ extension: `conv3d_fp8_forward(input, weight)` (no bias). Replace with your launch wrapper if needed.
- **`setup.py`** — Builds the extension with `torch.utils.cpp_extension` (ROCm/HIP or CUDA).
- **`conv3d_baseline.py`** — `CausalConv3dFP8`, `causal_conv3d_fp8`, and stage-0 shape checks. `padding=` / `bias=` on the module are API-only (unused).

## Build

From `conv3amd/ck_baseline/`:

```bash
pip install pybind11  # if needed
python setup.py build_ext --inplace
```

You should get `conv3d_my_impl*.so` in `ck_baseline/`.

## Use from Python

```python
import torch
from conv3d_baseline import CausalConv3dFP8

device = torch.device("cuda", 0)
dtype = torch.float8_e4m3fn
model = CausalConv3dFP8(128, 128, 3, padding=1).to(device, dtype)
x = torch.randn(1, 128, 4, 192, 256, device=device, dtype=torch.float32).to(dtype)
with torch.no_grad():
    y = model(x)
print(y.shape)  # (1, 128, 4, 192, 256)
```

## Replacing with your own kernel

1. Edit **`conv3d_my_kernel.cu`** (or HIP equivalent) and/or **`conv3d_my.cpp`**.
2. Keep **`conv3d_fp8_forward(torch::Tensor input, torch::Tensor weight)`** and the pybind name so Python does not change.
3. Rebuild: `python setup.py build_ext --inplace`.

## Use from conv3amd

```bash
cd conv3amd
python conv3amd.py --use-my-conv
```

(`--use-my-conv` / `--test-my-conv` require `float8_e4m3fn`; the script will warn and force it if needed.)

## Requirements

- PyTorch with `float8_e4m3fn`, built for **ROCm or CUDA** (must match your GPU).
  - **AMD:** install a ROCm wheel so `python -c "import torch; print(torch.version.hip)"` prints a version (not `None`). Otherwise `setup.py` picks the `.cu` path and **nvcc**, which will not target your GPU.
- **`pybind11`** headers (`pip install pybind11`); `./build_ck_baseline.sh` installs this if missing.
- C++17 compiler (+ **hipcc** for ROCm PyTorch or **nvcc** for CUDA PyTorch).
