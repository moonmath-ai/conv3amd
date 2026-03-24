# ck_baseline code review: `conv3amd.py` → Python → C++ → kernel

Traces how the baseline is invoked and how execution reaches the GPU kernel. **Only `float8_e4m3fn` is supported** (no BF16 in this package).

---

## 1. Entry from `conv3amd.py`

### When ck_baseline is used

- **`--use-my-conv`** — Uses `CausalConv3dFP8` with dtype `torch.float8_e4m3fn`.
- **`--test-my-conv`** — Compares nn `CausalConv3d` (BF16 reference) vs `CausalConv3dFP8` (extension); activations cast BF16 ↔ FP8 as in the script.

Calls into ck_baseline:

1. `model(x)` with `model` = `CausalConv3dFP8` → `forward(x)`.

---

## 2. FP8 path (end-to-end)

### 2.1 `conv3amd.py`

- Imports `CausalConv3dFP8` from `conv3d_baseline` when using `--use-my-conv`.

### 2.2 `conv3d_baseline.py` — `CausalConv3dFP8`

**Constructor:**

- Stage 0 only: 128→128, kernel 3×3×3; `padding=` / `bias=` ignored.
- `weight` is `float8_e4m3fn`.

**`forward(x)`:**

1. `_require_float8_e4m3fn(x, self.weight)`
2. `_verify_stage0_io(...)` — fixed `(B,128,4,192,320)` and weight shape
3. `return impl.conv3d_fp8_forward(x, self.weight)`

### 2.3 `conv3d_my.cpp` — `conv3d_fp8_forward`

- CUDA + dtype + 5D checks, then `launch_conv3d_my_kernel(input, weight)`.

### 2.4 `conv3d_my_kernel.cu` — `launch_conv3d_my_kernel`

- Asserts `Float8_e4m3fn` and 1 byte/element.
- Stub kernel: global→LDS touch, zeros output (replace with real conv).

---

## 3. Shared pieces

| Item | Where |
|------|--------|
| Bias | None in extension |
| Extension module | `conv3d_my_impl` |
| C++ entry | `conv3d_fp8_forward(input, weight)` |
| Kernel | `launch_conv3d_my_kernel` |

---

## 4. Plugging a custom kernel

Implement conv math inside `launch_conv3d_my_kernel` or call it from `conv3d_fp8_forward` after checks. Keep **`(input, weight)`** and **`float8_e4m3fn`** contract.
